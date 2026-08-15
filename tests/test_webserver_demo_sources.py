from __future__ import annotations

import io
import hashlib
import json
import stat
import threading
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from easyicu.webserver import demo_sources
from easyicu.webserver import demo_source_storage
from easyicu.webserver import jobs as job_store
from easyicu.webserver.app import app
from easyicu.webserver.jobs import Job, JobManager


def _zip(path: Path, entries: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def test_demo_catalog_is_fixed_safe_and_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "private-cache"))
    monkeypatch.setattr(
        demo_sources.source_store,
        "load_registry",
        lambda: {"ok": True, "sources": [], "active_path": None},
    )

    payload = demo_sources.demo_sources_catalog()

    assert payload["ok"] is True
    assert payload["cache"] == {
        "location": "user_cache",
        "override_env": "EASYICU_DEMO_CACHE_DIR",
    }
    assert [row["id"] for row in payload["sources"]] == [
        "mimic_iv_demo_v2_2",
        "eicu_demo_v2_0_1",
    ]
    mimic, eicu = payload["sources"]
    assert mimic["version"] == "2.2"
    assert mimic["database"] == "miiv"
    assert mimic["scope"]["patients"] == 100
    assert mimic["download"]["size_label"] == "15.5 MB"
    assert eicu["version"] == "2.0.1"
    assert eicu["database"] == "eicu_demo"
    assert eicu["scope"]["icu_stays"] == ">2500"
    assert eicu["download"]["size_label"] == "130.6 MB"
    assert all(
        row["provenance"]["provider"] == "PhysioNet"
        and row["provenance"]["download_url"].startswith("https://physionet.org/")
        and row["provenance"]["citation_url"].startswith("https://doi.org/")
        and row["provenance"]["license"]["url"].startswith("https://physionet.org/")
        and row["provenance"]["license"]["short_name"] == "ODbL 1.0"
        for row in payload["sources"]
    )
    serialized = json.dumps(payload)
    assert str(tmp_path) not in serialized
    assert all(row["status"]["state"] == "not_downloaded" for row in payload["sources"])


def test_safe_zip_extraction_rejects_traversal_and_symlink(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.zip"
    _zip(traversal, {"../escape.csv": b"not allowed"})
    with pytest.raises(demo_sources.DemoSourceError, match="Unsafe ZIP member"):
        demo_sources.safe_extract_zip(
            traversal,
            tmp_path / "traversal-out",
            max_uncompressed_bytes=1024,
        )
    assert not (tmp_path / "escape.csv").exists()

    linked = tmp_path / "linked.zip"
    info = zipfile.ZipInfo("dataset/link.csv")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(linked, "w") as archive:
        archive.writestr(info, "../../escape.csv")
    with pytest.raises(demo_sources.DemoSourceError, match="links are not allowed"):
        demo_sources.safe_extract_zip(
            linked,
            tmp_path / "linked-out",
            max_uncompressed_bytes=1024,
        )


def test_safe_zip_extraction_accepts_regular_files(tmp_path: Path) -> None:
    archive = tmp_path / "safe.zip"
    _zip(
        archive,
        {
            "official-demo/icu/icustays.csv": b"stay_id\n1\n",
            "official-demo/LICENSE.txt": b"ODbL",
        },
    )

    summary = demo_sources.safe_extract_zip(
        archive,
        tmp_path / "safe-out",
        max_uncompressed_bytes=1024,
    )

    assert summary == {"files": 2, "bytes": 14}
    assert (
        tmp_path / "safe-out" / "official-demo" / "icu" / "icustays.csv"
    ).read_bytes() == b"stay_id\n1\n"


def test_reused_archive_clears_exact_stale_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "cache"))
    canonical = demo_sources.get_source("eicu_demo_v2_0_1")
    seed = tmp_path / "seed.zip"
    _zip(seed, {"table.csv": b"a,b\n1,2\n"})
    source = replace(
        canonical,
        size_bytes=seed.stat().st_size,
        archive_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
    )
    paths = demo_source_storage.source_paths(source)
    paths.root.mkdir(parents=True)
    paths.archive.write_bytes(seed.read_bytes())
    partial, receipt = demo_source_storage._partial_paths(source, paths)
    partial.write_bytes(b"stale")
    demo_source_storage.write_marker(
        receipt,
        source,
        expected_size=source.size_bytes,
        strong_etag='"fixed-release"',
    )

    _, reused = demo_source_storage.download_archive(
        source,
        paths,
        Job("reuse", "demo-source-prepare"),
    )

    assert reused is True
    assert not partial.exists()
    assert not receipt.exists()


def test_stream_download_is_bounded_and_chunked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"0123456789"

    class Response(io.BytesIO):
        headers = {"Content-Length": str(len(payload))}

    reads: list[int] = []
    response = Response(payload)
    original_read = response.read

    def read(size: int = -1) -> bytes:
        reads.append(size)
        return original_read(size)

    monkeypatch.setattr(response, "read", read)
    job = Job("download", "test")

    received, digest = demo_sources._stream_response(
        response,
        tmp_path / "download.part",
        expected_size=len(payload),
        max_size=32,
        job=job,
    )

    assert received == len(payload)
    assert len(digest) == 64
    assert (tmp_path / "download.part").read_bytes() == payload
    assert reads and set(reads) == {1024 * 1024}
    assert job.events[-1]["phase"] == "download"
    assert job.events[-1]["download_rate_bps"] > 0
    assert job.events[-1]["eta_seconds"] == 0
    assert all("path" not in event for event in job.events)


def _zip_bytes() -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("official/icu/icustays.csv", "stay_id\n1\n2\n")
    return payload.getvalue()


class _DownloadResponse(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        status: int,
        headers: dict[str, str],
        final_url: str,
        fail_after_first_read: bool = False,
    ) -> None:
        super().__init__(payload)
        self.status = status
        self.headers = headers
        self._final_url = final_url
        self._fail_after_first_read = fail_after_first_read
        self._read_calls = 0

    def geturl(self) -> str:
        return self._final_url

    def read(self, size: int = -1) -> bytes:
        self._read_calls += 1
        if self._fail_after_first_read and self._read_calls > 1:
            raise OSError("simulated connection reset")
        return super().read(size)


def test_download_archive_resumes_verified_partial_with_range_and_if_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(demo_source_storage, "DOWNLOAD_CHUNK_BYTES", 8)
    payload = _zip_bytes()
    base = demo_sources.get_source("mimic_iv_demo_v2_2")
    source = replace(
        base,
        size_bytes=len(payload),
        max_download_bytes=len(payload) + 32,
        archive_sha256=hashlib.sha256(payload).hexdigest(),
    )
    paths = demo_source_storage.source_paths(source)
    final_url = "https://physionet.org/content/mimic-iv-demo/get-zip/2.2/"
    etag = '"demo-release-strong-etag"'
    calls: list[tuple[int, str | None]] = []

    def interrupted_response(
        _source,
        *,
        start_byte=0,
        validator=None,  # type: ignore[no-untyped-def]
    ):
        calls.append((start_byte, validator))
        return _DownloadResponse(
            payload,
            status=200,
            headers={
                "Content-Length": str(len(payload)),
                "Content-Type": "application/zip",
                "ETag": etag,
            },
            final_url=final_url,
            fail_after_first_read=True,
        )

    monkeypatch.setattr(
        demo_source_storage,
        "official_response",
        interrupted_response,
    )
    first_job = Job("first-range", "demo-source-prepare")
    with pytest.raises(OSError, match="connection reset"):
        demo_source_storage.download_archive(source, paths, first_job)

    partial, receipt = demo_source_storage._partial_paths(source, paths)
    assert partial.read_bytes() == payload[:8]
    assert demo_source_storage.read_marker(receipt, source)["strong_etag"] == etag
    interrupted_status = demo_source_storage.status_payload(source)
    assert interrupted_status["resume_available"] is True
    assert interrupted_status["partial_bytes"] == 8

    def resumed_response(
        _source,
        *,
        start_byte=0,
        validator=None,  # type: ignore[no-untyped-def]
    ):
        calls.append((start_byte, validator))
        return _DownloadResponse(
            payload[start_byte:],
            status=206,
            headers={
                "Content-Length": str(len(payload) - start_byte),
                "Content-Range": (
                    f"bytes {start_byte}-{len(payload) - 1}/{len(payload)}"
                ),
                "Content-Type": "application/zip",
                "ETag": etag,
            },
            final_url=final_url,
        )

    monkeypatch.setattr(
        demo_source_storage,
        "official_response",
        resumed_response,
    )
    second_job = Job("second-range", "demo-source-prepare")
    digest, reused = demo_source_storage.download_archive(source, paths, second_job)

    assert calls == [(0, None), (8, etag)]
    assert reused is False
    assert len(digest) == 64
    assert paths.archive.read_bytes() == payload
    assert not partial.exists()
    assert not receipt.exists()
    assert any(
        event.get("stage") == "resuming" and event.get("bytes_received") == 8
        for event in second_job.events
    )
    assert any(
        event.get("resume_from_bytes") == 8 and event.get("download_rate_bps", 0) > 0
        for event in second_job.events
    )


def test_download_archive_restarts_safely_when_range_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "cache"))
    payload = _zip_bytes()
    base = demo_sources.get_source("mimic_iv_demo_v2_2")
    source = replace(
        base,
        size_bytes=len(payload),
        max_download_bytes=len(payload) + 32,
        archive_sha256=hashlib.sha256(payload).hexdigest(),
    )
    paths = demo_source_storage.source_paths(source)
    paths.root.mkdir(parents=True)
    partial, receipt = demo_source_storage._partial_paths(source, paths)
    partial.write_bytes(payload[:7])
    old_etag = '"old-strong-etag"'
    demo_source_storage.write_marker(
        receipt,
        source,
        expected_size=len(payload),
        strong_etag=old_etag,
    )
    calls: list[tuple[int, str | None]] = []

    def full_response(
        _source,
        *,
        start_byte=0,
        validator=None,  # type: ignore[no-untyped-def]
    ):
        calls.append((start_byte, validator))
        return _DownloadResponse(
            payload,
            status=200,
            headers={
                "Content-Length": str(len(payload)),
                "Content-Type": "application/zip",
                "ETag": '"new-strong-etag"',
            },
            final_url=("https://physionet.org/content/mimic-iv-demo/get-zip/2.2/"),
        )

    monkeypatch.setattr(demo_source_storage, "official_response", full_response)
    job = Job("range-ignored", "demo-source-prepare")

    demo_source_storage.download_archive(source, paths, job)

    assert calls == [(7, old_etag)]
    assert paths.archive.read_bytes() == payload
    assert paths.archive.stat().st_size == len(payload)
    assert any(event.get("stage") == "restarting" for event in job.events)


def test_download_archive_rejects_misaligned_content_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "cache"))
    payload = _zip_bytes()
    base = demo_sources.get_source("mimic_iv_demo_v2_2")
    source = replace(
        base,
        size_bytes=len(payload),
        max_download_bytes=len(payload) + 32,
        archive_sha256=hashlib.sha256(payload).hexdigest(),
    )
    paths = demo_source_storage.source_paths(source)
    paths.root.mkdir(parents=True)
    partial, receipt = demo_source_storage._partial_paths(source, paths)
    partial.write_bytes(payload[:7])
    etag = '"demo-release-strong-etag"'
    demo_source_storage.write_marker(
        receipt,
        source,
        expected_size=len(payload),
        strong_etag=etag,
    )

    monkeypatch.setattr(
        demo_source_storage,
        "official_response",
        lambda _source, **_kwargs: _DownloadResponse(
            payload[7:],
            status=206,
            headers={
                "Content-Length": str(len(payload) - 7),
                "Content-Range": f"bytes 6-{len(payload) - 1}/{len(payload)}",
                "Content-Type": "application/zip",
                "ETag": etag,
            },
            final_url=("https://physionet.org/content/mimic-iv-demo/get-zip/2.2/"),
        ),
    )

    with pytest.raises(demo_sources.DemoSourceError, match="range does not match"):
        demo_source_storage.download_archive(
            source,
            paths,
            Job("bad-range", "demo-source-prepare"),
        )

    assert not partial.exists()
    assert not receipt.exists()


def test_prepare_runner_is_idempotent_and_registers_active_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "cache"))
    calls = {"download": 0, "convert": 0, "export": 0, "register": 0}

    def download(source, paths, job):  # type: ignore[no-untyped-def]
        calls["download"] += 1
        if not paths.archive.exists():
            paths.root.mkdir(parents=True, exist_ok=True)
            _zip(
                paths.archive,
                {"official/icu/icustays.csv": b"stay_id\n1\n"},
            )
        return "a" * 64, calls["download"] > 1

    def convert(source, paths, archive_sha256, job):  # type: ignore[no-untyped-def]
        calls["convert"] += 1
        if demo_sources._parquet_ready(paths, source):
            return {"converted": 1, "failed": 0, "skipped": 0, "total_files": 1}, True
        (paths.raw / "icustays.parquet").write_bytes(b"PAR1demoPAR1")
        demo_sources._write_marker(
            paths.converted_marker,
            source,
            archive_sha256=archive_sha256,
            conversion={
                "converted": 1,
                "failed": 0,
                "skipped": 0,
                "total_files": 1,
            },
        )
        job.emit({"type": "progress", "phase": "convert", "stage": "complete"})
        return {"converted": 1, "failed": 0, "skipped": 0, "total_files": 1}, False

    def export(source, paths, archive_sha256, job):  # type: ignore[no-untyped-def]
        calls["export"] += 1
        if demo_sources._export_ready(paths, source):
            return {
                "file_count": 19,
                "total_rows": 270,
                "manifest": "_manifest.json",
                "format": "parquet",
                "scope": "all_modules",
            }, True
        paths.export.mkdir(parents=True, exist_ok=True)
        (paths.export / "_manifest.json").write_text("{}", encoding="utf-8")
        summary = {
            "file_count": 19,
            "total_rows": 270,
            "manifest": "_manifest.json",
            "format": "parquet",
            "scope": "all_modules",
        }
        demo_sources._write_marker(
            paths.prepared_marker,
            source,
            archive_sha256=archive_sha256,
            export=summary,
        )
        job.emit({"type": "progress", "phase": "export", "stage": "complete"})
        return summary, False

    def register(path, **kwargs):  # type: ignore[no-untyped-def]
        calls["register"] += 1
        return {
            "ok": True,
            "sources": [{"ok": True, "path": path}],
            "active_path": path,
        }

    monkeypatch.setattr(demo_sources, "_download_archive", download)
    monkeypatch.setattr(demo_sources, "_convert_dataset", convert)
    monkeypatch.setattr(demo_sources, "_export_dataset", export)
    monkeypatch.setattr(demo_sources.source_store, "register_source", register)

    first_job = Job("first", "demo-source-prepare")
    first = demo_sources.make_prepare_runner("mimic_iv_demo_v2_2")(first_job)
    second_job = Job("second", "demo-source-prepare")
    second = demo_sources.make_prepare_runner("mimic_iv_demo_v2_2")(second_job)

    assert first["registered_source"] == {
        "ok": True,
        "active": True,
        "source_count": 1,
    }
    assert first["export"]["scope"] == "all_modules"
    assert first["reused"] is False
    assert second["reused"] is True
    assert set(second["reused_stages"]) == {"download", "extract", "convert", "export"}
    assert calls == {"download": 2, "convert": 2, "export": 2, "register": 2}
    phases = {
        event.get("phase")
        for event in first_job.events
        if event.get("type") == "progress"
    }
    assert {"extract", "convert", "export", "register"}.issubset(phases)
    serialized = json.dumps(first_job.events)
    assert str(tmp_path) not in serialized


def test_prepare_runner_preserves_structured_lower_layer_diagnostic_and_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        demo_sources,
        "_download_archive",
        lambda source, paths, job: ("a" * 64, False),
    )
    monkeypatch.setattr(
        demo_sources,
        "_extract_archive",
        lambda source, paths, archive_sha256, job: False,
    )
    monkeypatch.setattr(
        demo_sources,
        "_convert_dataset",
        lambda source, paths, archive_sha256, job: (
            {"converted": 1, "failed": 0, "skipped": 0, "total_files": 1},
            False,
        ),
    )
    lower = demo_sources.dataio.ExportCohortError(
        "column_metadata_primary_binding_missing",
        {
            "concepts": {"map", "hr"},
            "metadata": {
                "path": tmp_path / "typed-sidecar.json",
                "window": (0, 48),
                "score": float("nan"),
            },
        },
    )

    def fail_export(source, paths, archive_sha256, job):  # type: ignore[no-untyped-def]
        raise lower

    monkeypatch.setattr(demo_sources, "_export_dataset", fail_export)
    job = Job("structured", "demo-source-prepare")

    with pytest.raises(demo_sources.DemoSourceError) as exc_info:
        demo_sources.make_prepare_runner("mimic_iv_demo_v2_2")(job)

    expected = {
        "code": "demo_source_export_failed",
        "detail": {
            "phase": "export",
            "cause": {
                "type": "ExportCohortError",
                "structured": True,
                "code": "column_metadata_primary_binding_missing",
                "detail": {
                    "error": "column_metadata_primary_binding_missing",
                    "concepts": ["hr", "map"],
                    "metadata": {
                        "path": str(tmp_path / "typed-sidecar.json"),
                        "window": [0, 48],
                        "score": "NaN",
                    },
                },
            },
        },
    }
    assert exc_info.value.__cause__ is lower
    assert exc_info.value.code == expected["code"]
    assert exc_info.value.detail == expected["detail"]
    assert str(exc_info.value) == json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert json.loads(str(exc_info.value)) == expected


def test_prepare_stage_wraps_native_demo_error_for_job_and_preserves_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "cache"))
    raised: list[demo_sources.DemoSourceError] = []

    def fail_download(source, paths, job):  # type: ignore[no-untyped-def]
        error = demo_sources.DemoSourceError(
            "Official demo download size does not match the release"
        )
        raised.append(error)
        raise error

    monkeypatch.setattr(demo_sources, "_download_archive", fail_download)
    expected = {
        "code": "demo_source_download_failed",
        "detail": {
            "phase": "download",
            "cause": {
                "type": "DemoSourceError",
                "structured": True,
                "code": "demo_source_error",
                "detail": {
                    "message": (
                        "Official demo download size does not match the release"
                    )
                },
            },
        },
    }

    with pytest.raises(demo_sources.DemoSourceError) as exc_info:
        demo_sources.make_prepare_runner("mimic_iv_demo_v2_2")(
            Job("native-error", "demo-source-prepare")
        )

    assert exc_info.value.__cause__ is raised[0]
    assert json.loads(str(exc_info.value)) == expected

    manager = JobManager(max_completed=2)
    job = manager.submit(
        "demo-source-prepare",
        demo_sources.make_prepare_runner("mimic_iv_demo_v2_2"),
    )
    deadline = time.time() + 2
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.01)

    assert job.status == "failed"
    expected_json = json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert job.error == f"{expected['code']}: {expected_json}"
    assert json.loads(job.error.partition(": ")[2]) == expected


def test_prepare_stage_does_not_double_wrap_same_phase_diagnostic() -> None:
    diagnostic = demo_sources.DemoSourceError(
        "already wrapped",
        code="demo_source_export_failed",
        detail={
            "phase": "export",
            "cause": {
                "type": "ExportCohortError",
                "structured": True,
                "code": "column_metadata_primary_binding_missing",
                "detail": {"concept": "hr"},
            },
        },
    )

    def fail() -> None:
        raise diagnostic

    with pytest.raises(demo_sources.DemoSourceError) as exc_info:
        demo_sources._run_prepare_stage("export", fail)

    assert exc_info.value is diagnostic
    assert exc_info.value.__cause__ is None


def test_prepare_job_falls_back_to_stable_unstructured_diagnostic_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(
        demo_sources,
        "_download_archive",
        lambda source, paths, job: ("a" * 64, False),
    )
    monkeypatch.setattr(
        demo_sources,
        "_extract_archive",
        lambda source, paths, archive_sha256, job: False,
    )
    monkeypatch.setattr(
        demo_sources,
        "_convert_dataset",
        lambda source, paths, archive_sha256, job: (
            {"converted": 1, "failed": 0, "skipped": 0, "total_files": 1},
            False,
        ),
    )
    monkeypatch.setattr(
        demo_sources,
        "_export_dataset",
        lambda source, paths, archive_sha256, job: (
            {
                "file_count": 1,
                "total_rows": 1,
                "manifest": "_manifest.json",
                "format": "parquet",
                "scope": "all_modules",
            },
            False,
        ),
    )

    class UnstructuredLowerError(RuntimeError):
        error = 503
        detail = object()

    lower = UnstructuredLowerError("metadata writer unavailable")

    def fail_register(source, paths, job):  # type: ignore[no-untyped-def]
        raise lower

    monkeypatch.setattr(demo_sources, "_register_export", fail_register)
    manager = JobManager(max_completed=2)
    job = manager.submit(
        "demo-source-prepare",
        demo_sources.make_prepare_runner("eicu_demo_v2_0_1"),
    )
    deadline = time.time() + 2
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.01)

    expected = {
        "code": "demo_source_register_failed",
        "detail": {
            "phase": "register",
            "cause": {
                "type": "UnstructuredLowerError",
                "structured": False,
                "code": "unstructured_exception",
                "detail": {"message": "metadata writer unavailable"},
            },
        },
    }
    assert job.status == "failed"
    assert isinstance(job.error, str)
    expected_json = json.dumps(
        expected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert job.error == f"{expected['code']}: {expected_json}"
    assert json.loads(job.error.partition(": ")[2]) == expected
    assert job.snapshot()["error"] == job.error


def test_export_stage_uses_canonical_all_module_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EASYICU_DEMO_CACHE_DIR", str(tmp_path / "cache"))
    source = demo_sources.get_source("mimic_iv_demo_v2_2")
    paths = demo_sources._source_paths(source)
    paths.raw.mkdir(parents=True)
    captured: dict[str, object] = {}

    def make_export_runner(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)

        def run(job):  # type: ignore[no-untyped-def]
            job.emit(
                {
                    "type": "start",
                    "out_dir": str(tmp_path / "must-not-leak"),
                    "path": str(tmp_path / "must-not-leak"),
                }
            )
            return {
                "manifest": "_manifest.json",
                "file_count": 19,
                "total_rows": 270,
            }

        return run

    monkeypatch.setattr(demo_sources.dataio, "make_export_runner", make_export_runner)
    job = Job("export", "demo-source-prepare")

    summary, reused = demo_sources._export_dataset(
        source,
        paths,
        "a" * 64,
        job,
    )

    assert reused is False
    assert summary["scope"] == "all_modules"
    assert captured["modules"] is None
    assert captured["concepts"] is None
    assert captured["cohort"] is None
    assert captured["max_patients"] is None
    assert captured["export_format"] == "parquet"
    assert captured["create_run_subdir"] is False
    assert all("path" not in event and "out_dir" not in event for event in job.events)


def test_demo_source_owner_split_is_directional_and_facade_stays_thin() -> None:
    webserver = Path(__file__).parents[1] / "src" / "easyicu" / "webserver"
    facade = (webserver / "demo_sources.py").read_text(encoding="utf-8")
    contracts = (webserver / "demo_source_contracts.py").read_text(encoding="utf-8")
    storage = (webserver / "demo_source_storage.py").read_text(encoding="utf-8")
    prepare = (webserver / "demo_source_prepare.py").read_text(encoding="utf-8")

    assert len(facade.splitlines()) <= 130
    assert "import zipfile" not in facade
    assert "urlopen" not in facade
    assert "DataConverter" not in facade
    assert "from easyicu" not in contracts
    assert "import zipfile" in storage
    assert "build_opener" in storage
    assert "_PinnedPhysioNetRedirectHandler" in storage
    assert "DataConverter" not in storage
    assert "from easyicu.io.data_converter" in prepare
    assert "import zipfile" not in prepare
    assert "PrepareOperations(" in facade


def test_demo_source_routes_enforce_allowlist_and_submit_standard_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager(max_completed=10)
    monkeypatch.setattr(job_store, "MANAGER", manager)
    monkeypatch.setattr(
        demo_sources,
        "make_prepare_runner",
        lambda source_id: lambda job: {"source_id": source_id, "ok": True},
    )
    client = TestClient(app)

    catalog = client.get("/api/demo-sources")
    forbidden = client.post(
        "/api/jobs/demo-source-prepare",
        json={
            "source_id": "mimic_iv_demo_v2_2",
            "url": "https://attacker.invalid/demo.zip",
        },
    )
    unknown = client.post(
        "/api/jobs/demo-source-prepare",
        json={"source_id": "../../private"},
    )
    accepted = client.post(
        "/api/jobs/demo-source-prepare",
        json={"source_id": "mimic_iv_demo_v2_2"},
    )

    assert catalog.status_code == 200
    assert forbidden.status_code == 400
    assert forbidden.json()["detail"]["error"] == "invalid_demo_source_request"
    assert unknown.status_code == 400
    assert unknown.json()["detail"]["error"] == "unknown_demo_source"
    assert accepted.status_code == 200
    response = accepted.json()
    assert response["kind"] == "demo-source-prepare"
    deadline = time.time() + 2
    job = manager.get(response["job_id"])
    while job is not None and job.status == "running" and time.time() < deadline:
        time.sleep(0.01)
    assert job is not None
    assert job.status == "done"
    assert job.result == {"source_id": "mimic_iv_demo_v2_2", "ok": True}


def test_demo_source_submission_preserves_job_capacity_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager(max_completed=2, max_running=1)
    release = threading.Event()
    started = threading.Event()

    def blocking_runner(_job):  # type: ignore[no-untyped-def]
        started.set()
        release.wait(timeout=2)
        return {"ok": True}

    running = manager.submit("blocking", blocking_runner)
    assert started.wait(timeout=1)
    monkeypatch.setattr(job_store, "MANAGER", manager)
    monkeypatch.setattr(
        demo_sources,
        "make_prepare_runner",
        lambda source_id: lambda job: {"source_id": source_id},
    )

    response = TestClient(app).post(
        "/api/jobs/demo-source-prepare",
        json={"source_id": "eicu_demo_v2_0_1"},
    )
    release.set()
    deadline = time.time() + 2
    while running.status == "running" and time.time() < deadline:
        time.sleep(0.01)

    assert response.status_code == 429
    assert response.json()["detail"] == {
        "error": "job_capacity_exceeded",
        "running": 1,
        "max_running": 1,
        "reason": "Wait for a running local job to finish before retrying.",
    }
