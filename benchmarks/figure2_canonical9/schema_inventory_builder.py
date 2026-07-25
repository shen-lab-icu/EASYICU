"""Build a protected, metadata-only schema inventory for the exact full0717 export.

The inventory binds only file identities, manifest declarations, Parquet schema,
and row counts.  It does not read a patient row, infer a clinical concept, write
typed source metadata, or authorize materialization.  Its purpose is to give
data/identity reviewers one exact, reviewable input list before a future
source-attested typed inventory is prepared.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import stat
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from .identity_bridge_builder import (
    IdentityBridgeBuildError,
    full_export_content_sha256,
    full_export_file_identities,
)
from .source_attestation_contract import (
    FULL0717_EXPORT_CONTENT_SHA256,
    FULL0717_RUN_MANIFEST_SHA256,
)

SCHEMA_INVENTORY_SCHEMA = "easyicu.figure2_schema_inventory/1"
SCHEMA_INVENTORY_REF = "figure2_canonical9/full0717-schema-inventory/20260722-v1"
_EXPORT_LABEL = "full6_20260717"
_SOURCE_DIRECTORIES: tuple[tuple[str, str], ...] = (
    ("mimic_iv", "miiv"),
    ("mimic_iii", "mimic"),
    ("eicu", "eicu"),
    ("amsterdamumcdb", "aumc"),
    ("hirid", "hirid"),
    ("sicdb", "sic"),
)


class SchemaInventoryBuildError(ValueError):
    """The full0717 metadata-only review inventory cannot be built safely."""


@dataclasses.dataclass(frozen=True)
class SchemaInventoryBuildResult:
    """Non-sensitive output identities for one protected schema inventory."""

    output_root: Path
    inventory_path: Path
    inventory_sha256: str
    unlisted_member_count: int


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _safe_existing_directory(path: Path | str, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise SchemaInventoryBuildError(f"{label} must be an absolute real directory")
    return candidate.resolve(strict=True)


def _ensure_new_output_root(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink():
        raise SchemaInventoryBuildError(
            "output root must be an absolute non-symlink path"
        )
    if candidate.exists():
        raise SchemaInventoryBuildError("output root must not already exist")
    parent = candidate.parent
    if not parent.is_dir() or parent.is_symlink():
        raise SchemaInventoryBuildError("output root parent must be a real directory")
    return candidate


def _write_private_bytes(path: Path, content: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        stat.S_IRUSR | stat.S_IWUSR,
    )
    try:
        total = 0
        while total < len(content):
            written = os.write(descriptor, content[total:])
            if written <= 0:
                raise OSError("failed to write protected schema inventory")
            total += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaInventoryBuildError(
            f"cannot parse module manifest {path.name}"
        ) from exc
    if not isinstance(payload, dict):
        raise SchemaInventoryBuildError(
            f"module manifest {path.name} must be an object"
        )
    return payload


def _manifest_output(
    manifest: Mapping[str, Any], *, module: str
) -> dict[str, Any] | None:
    saved = manifest.get("saved")
    if not isinstance(saved, dict):
        raise SchemaInventoryBuildError(f"{module} manifest saved must be an object")
    if not saved:
        return None
    if len(saved) != 1:
        raise SchemaInventoryBuildError(
            f"{module} manifest must declare exactly one physical output"
        )
    output = next(iter(saved.values()))
    if not isinstance(output, dict):
        raise SchemaInventoryBuildError(
            f"{module} manifest saved output must be an object"
        )
    return output


def _schema_columns(parquet_path: Path) -> tuple[int, tuple[dict[str, object], ...]]:
    try:
        parquet = pq.ParquetFile(parquet_path)
    except Exception as exc:  # noqa: BLE001 - add file context for controlled review
        raise SchemaInventoryBuildError(
            f"cannot open {parquet_path.name} for schema-only inspection"
        ) from exc
    metadata = parquet.metadata
    if metadata is None:
        raise SchemaInventoryBuildError(f"{parquet_path.name} has no parquet metadata")
    return (
        int(metadata.num_rows),
        tuple(
            {
                "name": field.name,
                "arrow_type": str(field.type),
                "nullable": bool(field.nullable),
            }
            for field in parquet.schema_arrow
        ),
    )


def _module_record(
    *,
    export_root: Path,
    source_directory: str,
    manifest_path: Path,
    member_by_path: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], set[str]]:
    manifest_relative = manifest_path.relative_to(export_root).as_posix()
    module = manifest_path.name.removesuffix(".manifest.json")
    manifest = _load_manifest(manifest_path)
    declared_module = manifest.get("module")
    if declared_module != module:
        raise SchemaInventoryBuildError(
            f"{manifest_relative} module name does not match its file name"
        )
    manifest_identity = member_by_path.get(manifest_relative)
    if manifest_identity is None:
        raise SchemaInventoryBuildError(f"{manifest_relative} lacks a file identity")
    output = _manifest_output(manifest, module=module)
    record: dict[str, object] = {
        "module": module,
        "manifest": {
            "relative_path": manifest_relative,
            "sha256": manifest_identity["sha256"],
            "size_bytes": manifest_identity["size_bytes"],
        },
        "declared_errors": list(manifest.get("errors") or []),
        "declared_warnings": list(manifest.get("warnings") or []),
    }
    consumed = {manifest_relative}
    if output is None:
        record["state"] = "declared_empty"
        record["saved_output_names"] = []
        return record, consumed
    parquet_relative = f"{source_directory}/{module}.parquet"
    parquet_identity = member_by_path.get(parquet_relative)
    if parquet_identity is None:
        raise SchemaInventoryBuildError(
            f"{manifest_relative} declares an output but {parquet_relative} is absent"
        )
    rows, columns = _schema_columns(export_root / parquet_relative)
    declared_rows = output.get("rows")
    row_count_matches_manifest: bool | None = None
    manifest_missing_fields: list[str] = []
    manifest_invalid_fields: list[str] = []
    if declared_rows is None:
        manifest_missing_fields.append("rows")
    elif not isinstance(declared_rows, int) or isinstance(declared_rows, bool):
        manifest_invalid_fields.append("rows")
        declared_rows = None
    else:
        row_count_matches_manifest = rows == declared_rows
    concepts = output.get("concepts")
    if concepts is None:
        manifest_missing_fields.append("concepts")
    elif not isinstance(concepts, list) or not all(
        isinstance(value, str) and value for value in concepts
    ):
        manifest_invalid_fields.append("concepts")
        concepts = None
    record.update(
        {
            "state": "schema_observed_not_source_attested",
            "saved_output_names": sorted(manifest["saved"]),
            "manifest_completeness": (
                "complete"
                if not manifest_missing_fields and not manifest_invalid_fields
                else "partial_or_invalid"
            ),
            "manifest_missing_fields": manifest_missing_fields,
            "manifest_invalid_fields": manifest_invalid_fields,
            "manifest_declared_row_count": declared_rows,
            "row_count_matches_manifest": row_count_matches_manifest,
            "declared_concepts": list(concepts) if concepts is not None else None,
            "data_file": {
                "relative_path": parquet_relative,
                "sha256": parquet_identity["sha256"],
                "size_bytes": parquet_identity["size_bytes"],
                "row_count": rows,
                "columns": list(columns),
            },
        }
    )
    consumed.add(parquet_relative)
    return record, consumed


def build_schema_inventory(
    *, full_export_root: Path | str, output_root: Path | str
) -> SchemaInventoryBuildResult:
    """Build one protected metadata-only inventory for the exact full0717 bytes."""

    export_root = _safe_existing_directory(
        full_export_root, label="full0717 export root"
    )
    requested_output = _ensure_new_output_root(output_root)
    if export_root.name != _EXPORT_LABEL:
        raise SchemaInventoryBuildError(
            "schema inventory requires the full6_20260717 export"
        )
    try:
        members = full_export_file_identities(export_root)
        export_content_sha256 = full_export_content_sha256(export_root)
    except IdentityBridgeBuildError as exc:
        raise SchemaInventoryBuildError(
            "cannot fingerprint the full0717 export"
        ) from exc
    if export_content_sha256 != FULL0717_EXPORT_CONTENT_SHA256:
        raise SchemaInventoryBuildError(
            "full0717 content digest differs from the selected snapshot"
        )
    member_by_path = {str(item["relative_path"]): item for item in members}
    run_manifest = member_by_path.get("run_manifest.json")
    if run_manifest is None or run_manifest["sha256"] != FULL0717_RUN_MANIFEST_SHA256:
        raise SchemaInventoryBuildError(
            "full0717 run manifest does not match the review pin"
        )

    sources: list[dict[str, object]] = []
    consumed = {"run_manifest.json"}
    for source_id, source_directory in _SOURCE_DIRECTORIES:
        root = export_root / source_directory
        if root.is_symlink() or not root.is_dir():
            raise SchemaInventoryBuildError(
                f"full0717 is missing expected source directory {source_directory}"
            )
        manifests = sorted(root.glob("*.manifest.json"))
        if not manifests:
            raise SchemaInventoryBuildError(
                f"{source_directory} has no module manifests for review"
            )
        records: list[dict[str, object]] = []
        for manifest_path in manifests:
            record, used = _module_record(
                export_root=export_root,
                source_directory=source_directory,
                manifest_path=manifest_path,
                member_by_path=member_by_path,
            )
            records.append(record)
            consumed.update(used)
        sources.append(
            {
                "source_id": source_id,
                "export_directory": source_directory,
                "module_count": len(records),
                "modules": records,
            }
        )
    unlisted = [item for item in members if item["relative_path"] not in consumed]
    payload = {
        "schema_version": SCHEMA_INVENTORY_SCHEMA,
        "inventory_ref": SCHEMA_INVENTORY_REF,
        "historical_export": {
            "export_label": _EXPORT_LABEL,
            "export_content_sha256": export_content_sha256,
            "export_run_manifest_sha256": FULL0717_RUN_MANIFEST_SHA256,
        },
        "metadata_only": True,
        "source_attested": False,
        "real_run_authorized": False,
        "requires_unlisted_member_disposition": bool(unlisted),
        "sources": sources,
        "unlisted_members": unlisted,
    }
    temporary = (
        requested_output.parent / f".{requested_output.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary.mkdir(mode=0o700)
    try:
        inventory_path = temporary / "schema_inventory.json"
        inventory_raw = _canonical_json(payload)
        _write_private_bytes(inventory_path, inventory_raw)
        inventory_sha256 = hashlib.sha256(inventory_raw).hexdigest()
        receipt = {
            "schema_version": "easyicu.figure2_schema_inventory_receipt/1",
            "inventory_file": inventory_path.name,
            "inventory_sha256": inventory_sha256,
            "metadata_only": True,
            "source_attested": False,
            "real_run_authorized": False,
            "unlisted_member_count": len(unlisted),
        }
        _write_private_bytes(temporary / "build_receipt.json", _canonical_json(receipt))
        os.replace(temporary, requested_output)
        return SchemaInventoryBuildResult(
            output_root=requested_output,
            inventory_path=requested_output / inventory_path.name,
            inventory_sha256=inventory_sha256,
            unlisted_member_count=len(unlisted),
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-export-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_schema_inventory(
        full_export_root=args.full_export_root, output_root=args.output_root
    )
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "inventory_path": str(result.inventory_path),
                "inventory_sha256": result.inventory_sha256,
                "unlisted_member_count": result.unlisted_member_count,
                "metadata_only": True,
                "real_run_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())
