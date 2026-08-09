"""Build redistributable official-demo packs without leaking local paths.

This module owns the distribution boundary for the two allowlisted PhysioNet
demo releases.  It never accepts a caller-supplied URL and it does not handle
the credentialed full MIMIC-IV or eICU databases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable
import zipfile

from easyicu.webserver import demo_source_storage
from easyicu.webserver.demo_source_contracts import DemoSourcePaths, DemoSourceSpec
from easyicu.webserver.patient_drilldown.coverage import build_feature_coverage


PACK_SCHEMA = "easyicu.official_demo_release_pack/1"
_TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".yaml", ".yml", ".cff"}
_LOCAL_PATH_PATTERNS = (
    re.compile(r"/Users/[^/\s]+/"),
    re.compile(r"/home/[^/\s]+/"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
)
_ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


class DemoReleasePackError(RuntimeError):
    """A stable release-pack boundary failure."""


@dataclass(frozen=True)
class DemoReleaseReceipt:
    schema: str
    source_id: str
    source_version: str
    archive_path: str
    archive_sha256: str
    archive_size: int
    prepared_pack_path: str
    prepared_pack_sha256: str
    prepared_pack_size: int
    feature_summary: dict[str, int]


def build_release_pack(
    source: DemoSourceSpec,
    paths: DemoSourcePaths,
    output_dir: Path,
) -> DemoReleaseReceipt:
    """Build one verified official ZIP copy and one prepared EasyICU pack."""

    _validate_ready_source(source, paths)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_sha256 = _sha256_file(paths.archive)
    official_name = f"official-{source.archive_filename}"
    official_output = output_dir / official_name
    shutil.copyfile(paths.archive, official_output)
    if _sha256_file(official_output) != archive_sha256:
        raise DemoReleasePackError(
            "Copied official archive failed SHA-256 verification"
        )

    description = _release_export_description(paths.export)
    coverage = build_feature_coverage(paths.export, description)
    pack_name = f"easyicu-{source.id}-prepared-v1.zip"
    pack_output = output_dir / pack_name
    with tempfile.TemporaryDirectory(
        prefix=f".{source.id}-pack-", dir=output_dir
    ) as raw_staging:
        staging_root = Path(raw_staging) / f"easyicu-{source.id}-prepared-v1"
        export_root = staging_root / "export"
        export_root.mkdir(parents=True)
        _copy_sanitized_export(paths.export, export_root)
        license_path = _find_license(paths.raw)
        shutil.copyfile(license_path, staging_root / "LICENSE.txt")
        source_document = _source_document(
            source,
            archive_sha256=archive_sha256,
            coverage=coverage,
            prepared_marker=demo_source_storage.read_marker(
                paths.prepared_marker, source
            ),
        )
        _write_json(staging_root / "SOURCE.json", source_document)
        (staging_root / "NOTICE.md").write_text(_notice(source), encoding="utf-8")
        _write_pack_readme(staging_root / "README.md", source)
        _assert_no_local_paths(staging_root)
        _write_checksums(staging_root)
        _write_deterministic_zip(staging_root, pack_output)

    receipt = DemoReleaseReceipt(
        schema=PACK_SCHEMA,
        source_id=source.id,
        source_version=source.version,
        archive_path=str(official_output),
        archive_sha256=archive_sha256,
        archive_size=official_output.stat().st_size,
        prepared_pack_path=str(pack_output),
        prepared_pack_sha256=_sha256_file(pack_output),
        prepared_pack_size=pack_output.stat().st_size,
        feature_summary={
            str(key): int(value)
            for key, value in (coverage.get("summary") or {}).items()
        },
    )
    _write_json(output_dir / f"{source.id}-release-receipt.json", asdict(receipt))
    return receipt


def _validate_ready_source(source: DemoSourceSpec, paths: DemoSourcePaths) -> None:
    if not demo_source_storage.archive_ready(paths, source):
        raise DemoReleasePackError("The exact official archive is not ready")
    if not demo_source_storage.export_ready(paths, source):
        raise DemoReleasePackError("The prepared all-module export is not ready")
    if paths.archive.is_symlink() or paths.export.is_symlink():
        raise DemoReleasePackError("Release inputs must not be symbolic links")
    try:
        with zipfile.ZipFile(paths.archive) as archive:
            if archive.testzip() is not None:
                raise DemoReleasePackError("Official archive failed ZIP CRC validation")
    except zipfile.BadZipFile as error:
        raise DemoReleasePackError("Official archive is not a valid ZIP") from error


def _release_export_description(export_root: Path) -> dict[str, Any]:
    from easyicu.webserver import dataio

    description = dataio.describe_export_source(str(export_root))
    if not description.get("ok"):
        raise DemoReleasePackError("The prepared export inventory is unreadable")
    return description


def _copy_sanitized_export(source_root: Path, target_root: Path) -> None:
    for source in sorted(source_root.iterdir(), key=lambda path: path.name):
        if source.is_symlink() or not source.is_file():
            raise DemoReleasePackError(
                f"Unexpected non-file export member: {source.name}"
            )
        target = target_root / source.name
        if source.name == "_manifest.json":
            document = json.loads(source.read_text(encoding="utf-8"))
            document["data_path"] = "<official-demo-cache>/raw"
            export_folder = document.get("export_folder")
            if isinstance(export_folder, dict):
                export_folder["path"] = "<release-pack>/export"
            _write_json(target, document)
        elif source.name == "README.md":
            target.write_text(
                _sanitize_export_readme(source.read_text(encoding="utf-8")),
                encoding="utf-8",
            )
        else:
            shutil.copyfile(source, target)


def _sanitize_export_readme(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.startswith("- Source path:"):
            lines.append("- Source path: `<official-demo-cache>/raw`")
        else:
            lines.append(line)
    return "\n".join(lines).rstrip() + "\n"


def _find_license(raw_root: Path) -> Path:
    matches = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.name.lower() == "license.txt"
    )
    if not matches:
        raise DemoReleasePackError("The upstream ODbL LICENSE.txt is missing")
    license_path = matches[0]
    if (
        "open database license"
        not in license_path.read_text(encoding="utf-8", errors="replace").lower()
    ):
        raise DemoReleasePackError("The upstream license is not the expected ODbL text")
    return license_path


def _source_document(
    source: DemoSourceSpec,
    *,
    archive_sha256: str,
    coverage: dict[str, Any],
    prepared_marker: dict[str, Any] | None,
) -> dict[str, Any]:
    marker = dict(prepared_marker or {})
    marker.pop("updated_at", None)
    return {
        "schema": PACK_SCHEMA,
        "source": {
            "id": source.id,
            "title": source.title,
            "version": source.version,
            "database": source.database,
            "provider": "PhysioNet",
            "landing_page": source.landing_page,
            "download_url": source.download_url,
            "citation_url": source.citation_url,
            "attribution": source.attribution,
            "license": {
                "name": "Open Data Commons Open Database License v1.0",
                "short_name": "ODbL 1.0",
                "url": source.license_url,
                "terms_url": "https://opendatacommons.org/licenses/odbl/1-0/",
            },
        },
        "official_archive": {
            "filename": source.archive_filename,
            "size_bytes": source.size_bytes,
            "sha256": archive_sha256,
        },
        "easyicu_transform": {
            "scope": "all_19_catalog_modules",
            "format": "parquet",
            "prepared_marker": marker,
            "feature_coverage": coverage.get("summary") or {},
            "patient_rows_in_coverage_index": False,
        },
    }


def _notice(source: DemoSourceSpec) -> str:
    return f"""# Distribution notice

This package contains a transformed copy of the official, deidentified
**{source.title} (version {source.version})** demonstration database.

Source: {source.landing_page}

Required attribution: {source.attribution}

The database and this transformed database are distributed under the Open Data
Commons Open Database License v1.0 (ODbL 1.0). The complete license text is in
`LICENSE.txt`. EasyICU's software remains separately licensed under MIT.

This package does not contain the credentialed full MIMIC-IV or eICU database.
It contains only the openly downloadable official demonstration release.
"""


def _write_pack_readme(path: Path, source: DemoSourceSpec) -> None:
    path.write_text(
        f"""# EasyICU prepared official demo

Dataset: {source.title} {source.version}

The `export/` directory is an EasyICU all-module Parquet export. In Patient
Review, add that directory as a local export; the app reads only bounded,
pseudonymous review payloads and loads individual feature trajectories on
demand.

Verify the unpacked files with:

```bash
shasum -a 256 -c SHA256SUMS
```

See `SOURCE.json`, `NOTICE.md`, and `LICENSE.txt` before redistribution.
""",
        encoding="utf-8",
    )


def _assert_no_local_paths(root: Path) -> None:
    for path in _regular_files(root):
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in _LOCAL_PATH_PATTERNS):
            raise DemoReleasePackError(
                f"Release text contains a local absolute path: {path.name}"
            )


def _write_checksums(root: Path) -> None:
    rows = []
    for path in _regular_files(root):
        relative = path.relative_to(root).as_posix()
        if relative == "SHA256SUMS":
            continue
        rows.append(f"{_sha256_file(path)}  {relative}")
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _regular_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise DemoReleasePackError(f"Release member is a symlink: {path.name}")
        if path.is_file():
            yield path


def _write_deterministic_zip(source_root: Path, output_path: Path) -> None:
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with zipfile.ZipFile(
            temp_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for source in _regular_files(source_root):
                relative = source.relative_to(source_root.parent).as_posix()
                info = zipfile.ZipInfo(relative, date_time=_ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o100644 & 0xFFFF) << 16
                archive.writestr(info, source.read_bytes(), compresslevel=9)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_timestamp() -> str:
    """Return an ISO timestamp for external release notes, not pack bytes."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "DemoReleasePackError",
    "DemoReleaseReceipt",
    "PACK_SCHEMA",
    "build_release_pack",
    "build_timestamp",
]
