"""Private cache, secure download, and archive extraction for demo sources.

This owner is the only demo-source module allowed to resolve cache paths,
persist preparation markers, access the official archive URL, or unpack ZIP
members.  It depends only on the dependency-neutral contracts plus the source
registry needed to report readiness.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterator
from urllib.parse import urlparse
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from easyicu.webserver import sources as source_store
from easyicu.webserver.demo_source_contracts import (
    CACHE_ENV,
    MARKER_SCHEMA,
    DemoSourceError,
    DemoSourcePaths,
    DemoSourceSpec,
    check_cancelled,
)

DEFAULT_CACHE_ROOT = Path.home() / ".easyicu" / "demo_sources"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MAX_ZIP_ENTRIES = 5_000
CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def cache_root() -> Path:
    """Resolve the private cache root from the one supported override."""

    configured = os.getenv(CACHE_ENV)
    root = Path(configured).expanduser() if configured else DEFAULT_CACHE_ROOT
    return root.resolve()


def source_paths(source: DemoSourceSpec) -> DemoSourcePaths:
    """Compile all private paths for one immutable allowlist entry."""

    root = cache_root() / source.id
    raw = root / "raw"
    export = root / "export"
    return DemoSourcePaths(
        root=root,
        archive=root / source.archive_filename,
        raw=raw,
        export=export,
        extracted_marker=raw / ".easyicu-demo-extracted.json",
        converted_marker=raw / ".easyicu-demo-converted.json",
        prepared_marker=export / ".easyicu-demo-prepared.json",
    )


def now() -> str:
    """Return the canonical marker timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_marker(
    path: Path,
    source: DemoSourceSpec,
) -> dict[str, Any] | None:
    """Read a small, regular marker belonging to the exact release."""

    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema") != MARKER_SCHEMA
        or payload.get("source_id") != source.id
        or payload.get("version") != source.version
    ):
        return None
    return payload


def write_marker(
    path: Path,
    source: DemoSourceSpec,
    **payload: Any,
) -> None:
    """Atomically persist one private, release-bound readiness marker."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    document = {
        "schema": MARKER_SCHEMA,
        "source_id": source.id,
        "version": source.version,
        "updated_at": now(),
        **payload,
    }
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        tmp.replace(path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def archive_ready(paths: DemoSourcePaths, source: DemoSourceSpec) -> bool:
    """Return whether the cached archive is a regular file of the fixed size."""

    try:
        return (
            not paths.archive.is_symlink()
            and paths.archive.is_file()
            and paths.archive.stat().st_size == source.size_bytes
        )
    except OSError:
        return False


def raw_ready(paths: DemoSourcePaths, source: DemoSourceSpec) -> bool:
    """Return whether secure extraction completed for this release."""

    return (
        paths.raw.is_dir()
        and not paths.raw.is_symlink()
        and read_marker(paths.extracted_marker, source) is not None
    )


def parquet_ready(paths: DemoSourcePaths, source: DemoSourceSpec) -> bool:
    """Return whether canonical table conversion completed."""

    return raw_ready(paths, source) and (
        read_marker(paths.converted_marker, source) is not None
    )


def export_ready(paths: DemoSourcePaths, source: DemoSourceSpec) -> bool:
    """Return whether the all-module export and its marker both exist."""

    manifest = paths.export / "_manifest.json"
    return (
        paths.export.is_dir()
        and not paths.export.is_symlink()
        and manifest.is_file()
        and not manifest.is_symlink()
        and read_marker(paths.prepared_marker, source) is not None
    )


def registry_state(paths: DemoSourcePaths) -> tuple[bool, bool]:
    """Read registration/active flags without leaking or trusting cache paths."""

    try:
        registry = source_store.load_registry()
        target = str(paths.export.resolve())
        registered = any(
            bool(item.get("ok"))
            and str(Path(str(item.get("path") or "")).expanduser().resolve()) == target
            for item in registry.get("sources") or []
            if isinstance(item, dict)
        )
        active = (
            registered
            and str(Path(str(registry.get("active_path") or "")).expanduser().resolve())
            == target
        )
        return registered, active
    except Exception:  # noqa: BLE001 - status must remain available if registry is bad.
        return False, False


def status_payload(source: DemoSourceSpec) -> dict[str, Any]:
    """Return path-free cache readiness for one source."""

    paths = source_paths(source)
    partial_bytes = _resumable_partial_bytes(source, paths)
    is_archive_ready = archive_ready(paths, source)
    is_raw_ready = raw_ready(paths, source)
    is_parquet_ready = parquet_ready(paths, source)
    is_export_ready = export_ready(paths, source)
    registered, active = registry_state(paths) if is_export_ready else (False, False)
    if is_export_ready:
        state = "prepared"
    elif is_parquet_ready:
        state = "converted"
    elif is_raw_ready or is_archive_ready:
        state = "downloaded"
    else:
        state = "not_downloaded"
    marker = read_marker(paths.prepared_marker, source) if is_export_ready else None
    return {
        "state": state,
        "archive_ready": is_archive_ready,
        "raw_ready": is_raw_ready,
        "parquet_ready": is_parquet_ready,
        "export_ready": is_export_ready,
        "registered": registered,
        "active": active,
        "prepared_at": (marker or {}).get("updated_at"),
        "resume_available": partial_bytes > 0,
        "partial_bytes": partial_bytes,
    }


def _canonical_download_path(source: DemoSourceSpec) -> str:
    """Return the one PhysioNet redirect path allowed for this release."""

    landing = urlparse(source.landing_page).path.rstrip("/")
    prefix, _, version = landing.rpartition("/")
    return f"{prefix}/get-zip/{version}/"


def validate_official_response_url(
    raw_url: str,
    source: DemoSourceSpec | None = None,
) -> None:
    """Reject redirects outside the pinned HTTPS PhysioNet release."""

    parsed = urlparse(raw_url)
    try:
        port = parsed.port
    except ValueError as error:
        raise DemoSourceError("PhysioNet download used an invalid port") from error
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != "physionet.org"
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise DemoSourceError("PhysioNet download redirected to an untrusted host")
    if source is None:
        return
    allowed_paths = {
        urlparse(source.download_url).path,
        _canonical_download_path(source),
    }
    if parsed.path not in allowed_paths or parsed.query or parsed.fragment:
        raise DemoSourceError("PhysioNet download redirected outside the pinned release")


def validate_download_response_url(
    raw_url: str,
    source: DemoSourceSpec,
) -> None:
    """Allow only the exact PhysioNet release or its pinned GitHub mirror."""

    parsed = urlparse(raw_url)
    hostname = (parsed.hostname or "").lower()
    if hostname == "physionet.org":
        validate_official_response_url(raw_url, source)
        return
    try:
        port = parsed.port
    except ValueError as error:
        raise DemoSourceError("Demo mirror download used an invalid port") from error
    if (
        parsed.scheme != "https"
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise DemoSourceError("Demo mirror redirected to an untrusted host")
    if hostname == "github.com":
        mirror = urlparse(source.mirror_url or "")
        if (
            not source.mirror_url
            or parsed.path != mirror.path
            or parsed.query
            or parsed.fragment
        ):
            raise DemoSourceError("GitHub mirror redirected outside the pinned release")
        return
    if hostname == "release-assets.githubusercontent.com":
        # GitHub signs opaque release-asset URLs.  The allowlisted source size
        # and pinned SHA-256 remain the final immutable-content boundary.
        if not source.mirror_url or not parsed.path.startswith("/github-production-release-asset/"):
            raise DemoSourceError("GitHub mirror redirected outside release assets")
        return
    raise DemoSourceError("Demo download redirected to an untrusted host")


class _PinnedPhysioNetRedirectHandler(HTTPRedirectHandler):
    """Validate every official-or-mirror redirect before opening it."""

    def __init__(self, source: DemoSourceSpec) -> None:
        self._source = source
        super().__init__()

    def redirect_request(  # type: ignore[override]
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        validate_download_response_url(newurl, self._source)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def official_response(
    source: DemoSourceSpec,
    *,
    start_byte: int = 0,
    validator: str | None = None,
):
    """Open the pinned GitHub mirror, falling back to the PhysioNet release."""

    if start_byte < 0 or start_byte >= source.size_bytes:
        raise DemoSourceError("Official demo download resume offset is invalid")
    headers = {
        "User-Agent": "EasyICU/official-demo-preparer",
        "Accept-Encoding": "identity",
    }
    if start_byte:
        if not validator or validator.startswith("W/"):
            raise DemoSourceError("Official demo download has no strong resume validator")
        headers["Range"] = f"bytes={start_byte}-{source.size_bytes - 1}"
        headers["If-Range"] = validator
    opener = build_opener(_PinnedPhysioNetRedirectHandler(source))
    urls = [url for url in (source.mirror_url, source.download_url) if url]
    last_error: Exception | None = None
    for url in urls:
        request = Request(url, headers=headers)
        try:
            return opener.open(request, timeout=60)  # noqa: S310 - pinned URLs.
        except (URLError, TimeoutError, OSError) as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise DemoSourceError("Official demo release has no download transport")


def _header(response: BinaryIO, name: str) -> str | None:
    """Read and normalize one HTTP response header."""

    value = getattr(response, "headers", {}).get(name)
    return str(value).strip() if value is not None else None


def _declared_length(response: BinaryIO) -> int | None:
    """Parse Content-Length strictly when the server supplies it."""

    raw = _header(response, "Content-Length")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as error:
        raise DemoSourceError(
            "Official demo download has an unsafe content length"
        ) from error


def _response_status(response: BinaryIO) -> int | None:
    """Read the final HTTP status from urllib and test doubles."""

    raw = getattr(response, "status", None)
    if raw is None:
        getcode = getattr(response, "getcode", None)
        raw = getcode() if callable(getcode) else None
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _strong_etag(response: BinaryIO) -> str | None:
    """Return a strong entity tag suitable for If-Range."""

    value = _header(response, "ETag")
    if not value or value.startswith("W/"):
        return None
    return value


def _validate_download_response(
    response: BinaryIO,
    source: DemoSourceSpec,
    *,
    start_byte: int,
    validator: str | None,
) -> bool:
    """Validate a full or ranged response; return whether it may be appended."""

    encoding = (_header(response, "Content-Encoding") or "identity").lower()
    if encoding != "identity":
        raise DemoSourceError("Official demo download used unsupported content encoding")
    content_type = (_header(response, "Content-Type") or "").split(";", 1)[0].lower()
    if content_type and content_type not in {
        "application/zip",
        "application/octet-stream",
    }:
        raise DemoSourceError("Official demo response is not a ZIP download")

    status = _response_status(response)
    if start_byte and status == 200:
        return False
    if start_byte:
        if status != 206:
            raise DemoSourceError("Official demo resume response status is invalid")
        match = CONTENT_RANGE_RE.fullmatch(_header(response, "Content-Range") or "")
        if not match:
            raise DemoSourceError("Official demo resume range is invalid")
        first, last, total = (int(value) for value in match.groups())
        if (
            first != start_byte
            or last != source.size_bytes - 1
            or total != source.size_bytes
        ):
            raise DemoSourceError("Official demo resume range does not match the release")
        if not validator or _strong_etag(response) != validator:
            raise DemoSourceError("Official demo resume validator changed")
        return True
    if status not in {None, 200}:
        raise DemoSourceError("Official demo download response status is invalid")
    return False


def stream_response(
    response: BinaryIO,
    destination: Path,
    *,
    expected_size: int,
    max_size: int,
    job: Any,
    resume_from: int = 0,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[int, str]:
    """Stream a full or verified ranged response under fixed byte bounds."""

    if expected_size < 1 or expected_size > max_size:
        raise DemoSourceError("Official demo download size exceeds its release limit")
    if resume_from < 0 or resume_from >= expected_size:
        raise DemoSourceError("Official demo download resume offset is invalid")
    remaining = expected_size - resume_from
    declared = _declared_length(response)
    if declared is not None and (declared < 1 or declared > remaining):
        raise DemoSourceError("Official demo download has an unsafe content length")
    if declared is not None and declared != remaining:
        raise DemoSourceError("Official demo download size does not match the release")

    digest = hashlib.sha256()
    if resume_from:
        try:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.stat().st_size != resume_from
            ):
                raise DemoSourceError("Official demo partial download state is unsafe")
        except OSError as error:
            raise DemoSourceError(
                "Official demo partial download state is unavailable"
            ) from error
        with destination.open("rb") as existing:
            for chunk in iter(lambda: existing.read(DOWNLOAD_CHUNK_BYTES), b""):
                digest.update(chunk)
        mode = "ab"
    else:
        mode = "xb"

    received = resume_from
    response_received = 0
    started = clock()
    with destination.open(mode) as handle:
        if not resume_from:
            os.chmod(destination, 0o600)
        while True:
            check_cancelled(job, "download")
            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            received += len(chunk)
            response_received += len(chunk)
            if received > max_size:
                raise DemoSourceError("Official demo download exceeded its size limit")
            digest.update(chunk)
            handle.write(chunk)
            elapsed = max(clock() - started, 0.001)
            rate = max(1, int(response_received / elapsed))
            eta = max(0, int(round((expected_size - received) / rate)))
            job.emit(
                {
                    "type": "progress",
                    "phase": "download",
                    "stage": "streaming",
                    "bytes_received": received,
                    "bytes_total": expected_size,
                    "resume_from_bytes": resume_from,
                    "download_rate_bps": rate,
                    "eta_seconds": eta,
                }
            )
        handle.flush()
        os.fsync(handle.fileno())
    if received != expected_size or (
        declared is not None and response_received != declared
    ):
        raise DemoSourceError("Official demo download ended before the expected size")
    return received, digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a cached archive without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _partial_paths(
    source: DemoSourceSpec,
    paths: DemoSourcePaths,
) -> tuple[Path, Path]:
    """Return deterministic private paths for resumable bytes and their receipt."""

    return (
        paths.root / f".{source.archive_filename}.part",
        paths.root / f".{source.archive_filename}.part.json",
    )


def _resumable_partial_bytes(
    source: DemoSourceSpec,
    paths: DemoSourcePaths,
) -> int:
    """Return only a safe, validator-bound partial byte count for public status."""

    partial, receipt = _partial_paths(source, paths)
    try:
        if partial.is_symlink() or not partial.is_file():
            return 0
        size = partial.stat().st_size
    except OSError:
        return 0
    if size <= 0 or size >= source.size_bytes:
        return 0
    marker = read_marker(receipt, source) or {}
    validator = marker.get("strong_etag")
    if (
        marker.get("expected_size") != source.size_bytes
        or not isinstance(validator, str)
        or not validator
        or validator.startswith("W/")
    ):
        return 0
    return size


def _clear_partial(partial: Path, receipt: Path) -> None:
    """Remove only the two exact private partial files, rejecting directories."""

    for target in (partial, receipt):
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            raise DemoSourceError("Official demo partial download path is unsafe")


def _resume_state(
    source: DemoSourceSpec,
    partial: Path,
    receipt: Path,
) -> tuple[int, str | None]:
    """Return a verified resumable offset and strong validator."""

    if partial.is_symlink():
        raise DemoSourceError("Official demo partial download path is unsafe")
    if not partial.exists():
        if receipt.is_symlink() or receipt.is_file():
            receipt.unlink()
        elif receipt.exists():
            raise DemoSourceError("Official demo partial receipt path is unsafe")
        return 0, None
    if not partial.is_file():
        raise DemoSourceError("Official demo partial download path is unsafe")
    try:
        size = partial.stat().st_size
    except OSError as error:
        raise DemoSourceError(
            "Official demo partial download state is unavailable"
        ) from error
    if size <= 0 or size > source.size_bytes:
        _clear_partial(partial, receipt)
        return 0, None
    if size == source.size_bytes:
        return size, None

    marker = read_marker(receipt, source) or {}
    validator = marker.get("strong_etag")
    if (
        marker.get("expected_size") != source.size_bytes
        or not isinstance(validator, str)
        or not validator
        or validator.startswith("W/")
    ):
        _clear_partial(partial, receipt)
        return 0, None
    return size, validator


def _validate_zip_archive(path: Path) -> None:
    """Validate the completed archive before it may replace the cache target."""

    if not zipfile.is_zipfile(path):
        raise DemoSourceError("Official demo response is not a valid ZIP archive")
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
    if bad_member:
        raise DemoSourceError("Official demo ZIP failed its CRC integrity check")


def _finish_partial_archive(
    source: DemoSourceSpec,
    paths: DemoSourcePaths,
    partial: Path,
    receipt: Path,
    job: Any,
    *,
    resume_from: int,
) -> str:
    """Integrity-check and atomically promote a completed partial archive."""

    try:
        _validate_zip_archive(partial)
        digest = sha256_file(partial)
        if source.archive_sha256 and digest != source.archive_sha256:
            raise DemoSourceError(
                "Official demo archive SHA-256 does not match the pinned release"
            )
    except Exception:
        _clear_partial(partial, receipt)
        raise
    partial.replace(paths.archive)
    if receipt.is_symlink() or receipt.is_file():
        receipt.unlink()
    elif receipt.exists():
        raise DemoSourceError("Official demo partial receipt path is unsafe")
    job.emit(
        {
            "type": "progress",
            "phase": "download",
            "stage": "complete",
            "bytes_received": source.size_bytes,
            "bytes_total": source.size_bytes,
            "resume_from_bytes": resume_from,
            "sha256": digest,
        }
    )
    return digest


def download_archive(
    source: DemoSourceSpec,
    paths: DemoSourcePaths,
    job: Any,
) -> tuple[str, bool]:
    """Download and validate the exact official archive, or reuse it."""

    if archive_ready(paths, source):
        try:
            _validate_zip_archive(paths.archive)
            digest = sha256_file(paths.archive)
            if source.archive_sha256 and digest != source.archive_sha256:
                raise DemoSourceError(
                    "Cached demo archive SHA-256 does not match the pinned release"
                )
        except DemoSourceError:
            pass
        else:
            partial, receipt = _partial_paths(source, paths)
            _clear_partial(partial, receipt)
            job.emit(
                {
                    "type": "progress",
                    "phase": "download",
                    "stage": "reused",
                    "bytes_received": source.size_bytes,
                    "bytes_total": source.size_bytes,
                }
            )
            return digest, True

    paths.root.mkdir(parents=True, exist_ok=True, mode=0o700)
    partial, receipt = _partial_paths(source, paths)
    offset, validator = _resume_state(source, partial, receipt)
    if offset == source.size_bytes:
        digest = _finish_partial_archive(
            source,
            paths,
            partial,
            receipt,
            job,
            resume_from=offset,
        )
        return digest, False

    job.emit(
        {
            "type": "progress",
            "phase": "download",
            "stage": "resuming" if offset else "starting",
            "bytes_received": offset,
            "bytes_total": source.size_bytes,
            "resume_from_bytes": offset,
        }
    )

    try:
        with official_response(
            source,
            start_byte=offset,
            validator=validator,
        ) as response:
            final_url = getattr(response, "geturl", lambda: source.download_url)()
            validate_download_response_url(str(final_url), source)
            try:
                may_append = _validate_download_response(
                    response,
                    source,
                    start_byte=offset,
                    validator=validator,
                )
            except DemoSourceError:
                if offset:
                    _clear_partial(partial, receipt)
                raise
            if offset and not may_append:
                _clear_partial(partial, receipt)
                offset = 0
                validator = None
                job.emit(
                    {
                        "type": "progress",
                        "phase": "download",
                        "stage": "restarting",
                        "bytes_received": 0,
                        "bytes_total": source.size_bytes,
                        "resume_from_bytes": 0,
                    }
                )
            if not may_append:
                validator = _strong_etag(response)
                if validator:
                    write_marker(
                        receipt,
                        source,
                        expected_size=source.size_bytes,
                        strong_etag=validator,
                    )
            stream_response(
                response,
                partial,
                expected_size=source.size_bytes,
                max_size=source.max_download_bytes,
                job=job,
                resume_from=offset,
            )
    except Exception:
        marker = read_marker(receipt, source) or {}
        resumable = (
            isinstance(marker.get("strong_etag"), str)
            and not str(marker.get("strong_etag")).startswith("W/")
        )
        if not resumable:
            _clear_partial(partial, receipt)
        raise

    digest = _finish_partial_archive(
        source,
        paths,
        partial,
        receipt,
        job,
        resume_from=offset,
    )
    return digest, False


def safe_zip_members(
    archive: zipfile.ZipFile,
    *,
    max_uncompressed_bytes: int,
) -> Iterator[tuple[zipfile.ZipInfo, PurePosixPath]]:
    """Yield normalized regular ZIP members under aggregate safety bounds."""

    infos = archive.infolist()
    if len(infos) > MAX_ZIP_ENTRIES:
        raise DemoSourceError("Official demo ZIP contains too many entries")
    total = 0
    seen: set[str] = set()
    for info in infos:
        raw_name = info.filename
        if "\x00" in raw_name:
            raise DemoSourceError("ZIP member contains a null byte")
        normalized = raw_name.replace("\\", "/")
        member = PurePosixPath(normalized)
        parts = tuple(part for part in member.parts if part not in {"", "."})
        if not parts or member.is_absolute() or ".." in parts or parts[0].endswith(":"):
            raise DemoSourceError(f"Unsafe ZIP member path: {raw_name}")
        clean = PurePosixPath(*parts)
        clean_key = clean.as_posix().casefold()
        if clean_key in seen:
            raise DemoSourceError(f"Duplicate ZIP member path: {raw_name}")
        seen.add(clean_key)

        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if stat.S_ISLNK(unix_mode):
            raise DemoSourceError(f"ZIP links are not allowed: {raw_name}")
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise DemoSourceError(f"Unsupported ZIP member type: {raw_name}")
        if info.flag_bits & 0x1:
            raise DemoSourceError(f"Encrypted ZIP members are not allowed: {raw_name}")
        if info.file_size < 0 or info.file_size > max_uncompressed_bytes:
            raise DemoSourceError(f"ZIP member exceeds extraction limit: {raw_name}")
        total += info.file_size
        if total > max_uncompressed_bytes:
            raise DemoSourceError("Official demo ZIP exceeds its extraction limit")
        if (
            info.compress_size > 0
            and info.file_size > 0
            and info.file_size / info.compress_size > 1_000
        ):
            raise DemoSourceError(f"ZIP member compression ratio is unsafe: {raw_name}")
        yield info, clean


def safe_extract_zip(
    archive_path: str | Path,
    destination: str | Path,
    *,
    max_uncompressed_bytes: int,
) -> dict[str, int]:
    """Extract a ZIP while rejecting traversal, links, special files and bombs."""

    archive_path = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    base = destination.resolve()
    extracted_files = 0
    extracted_bytes = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = list(
            safe_zip_members(
                archive,
                max_uncompressed_bytes=max_uncompressed_bytes,
            )
        )
        for info, member in members:
            target = (base / Path(*member.parts)).resolve()
            try:
                target.relative_to(base)
            except ValueError as exc:
                raise DemoSourceError(
                    f"Unsafe ZIP member path: {info.filename}"
                ) from exc
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
                continue
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if target.exists() or target.is_symlink():
                raise DemoSourceError(
                    f"Duplicate ZIP extraction target: {info.filename}"
                )
            written = 0
            with archive.open(info, "r") as source, target.open("xb") as output:
                os.chmod(target, 0o600)
                while True:
                    chunk = source.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    extracted_bytes += len(chunk)
                    if (
                        written > info.file_size
                        or extracted_bytes > max_uncompressed_bytes
                    ):
                        raise DemoSourceError("ZIP expanded beyond its declared limit")
                    output.write(chunk)
            if written != info.file_size:
                raise DemoSourceError(f"ZIP member size mismatch: {info.filename}")
            extracted_files += 1
    return {"files": extracted_files, "bytes": extracted_bytes}


def replace_directory_atomically(staged: Path, destination: Path) -> None:
    """Swap a staged directory into place, restoring the old one on failure."""

    stale: Path | None = None
    if destination.exists() or destination.is_symlink():
        stale = destination.parent / f".{destination.name}.stale-{uuid.uuid4().hex}"
        destination.replace(stale)
    try:
        staged.replace(destination)
    except Exception:
        if stale is not None and stale.exists() and not destination.exists():
            stale.replace(destination)
        raise
    if stale is not None:
        if stale.is_symlink() or stale.is_file():
            stale.unlink()
        else:
            shutil.rmtree(stale)


def extract_archive(
    source: DemoSourceSpec,
    paths: DemoSourcePaths,
    archive_sha256: str,
    job: Any,
) -> bool:
    """Securely extract an archive into the private cache, or reuse it."""

    if raw_ready(paths, source):
        job.emit({"type": "progress", "phase": "extract", "stage": "reused"})
        return True
    check_cancelled(job, "extract")
    staged = paths.root / f".raw-{uuid.uuid4().hex}.tmp"
    job.emit({"type": "progress", "phase": "extract", "stage": "starting"})
    try:
        summary = safe_extract_zip(
            paths.archive,
            staged,
            max_uncompressed_bytes=source.max_uncompressed_bytes,
        )
        write_marker(
            staged / paths.extracted_marker.name,
            source,
            archive_sha256=archive_sha256,
            extracted_files=summary["files"],
            extracted_bytes=summary["bytes"],
        )
        replace_directory_atomically(staged, paths.raw)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    job.emit(
        {
            "type": "progress",
            "phase": "extract",
            "stage": "complete",
            **summary,
        }
    )
    return False


__all__ = [
    "DEFAULT_CACHE_ROOT",
    "DOWNLOAD_CHUNK_BYTES",
    "MAX_ZIP_ENTRIES",
    "archive_ready",
    "cache_root",
    "download_archive",
    "export_ready",
    "extract_archive",
    "official_response",
    "parquet_ready",
    "raw_ready",
    "read_marker",
    "registry_state",
    "replace_directory_atomically",
    "safe_extract_zip",
    "safe_zip_members",
    "sha256_file",
    "source_paths",
    "status_payload",
    "stream_response",
    "validate_download_response_url",
    "validate_official_response_url",
    "write_marker",
]
