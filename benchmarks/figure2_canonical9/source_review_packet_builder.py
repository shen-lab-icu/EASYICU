"""Build a small, non-authorizing review packet from the full0717 inventory.

The packet deliberately consumes the protected *metadata-only* inventory rather
than the clinical export.  It groups the exact outstanding review work by
source, partial manifest, and unlisted member so data/identity/methods owners
can sign a bounded checklist.  It cannot add semantic source metadata, attest a
source, authorize P4, or launch a Canonical9 run.
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .source_attestation_contract import (
    FULL0717_EXPORT_CONTENT_SHA256,
    FULL0717_RUN_MANIFEST_SHA256,
)

SOURCE_REVIEW_PACKET_SCHEMA = "easyicu.figure2_source_review_packet/1"
SOURCE_REVIEW_PACKET_REF = (
    "figure2_canonical9/full0717-source-review-packet/20260722-v1"
)
FULL0717_SCHEMA_INVENTORY_SHA256 = (
    "e3ad266bc7f2ce7896d0b9361e5c9d0cf1c8bd9498d06adf24b95e9fe39b1474"
)
_SCHEMA_INVENTORY_SCHEMA = "easyicu.figure2_schema_inventory/1"
_SCHEMA_INVENTORY_REF = "figure2_canonical9/full0717-schema-inventory/20260722-v1"
_MAX_INVENTORY_BYTES = 1024 * 1024
_SOURCE_IDS = (
    "mimic_iv",
    "mimic_iii",
    "eicu",
    "amsterdamumcdb",
    "hirid",
    "sicdb",
)


class SourceReviewPacketBuildError(ValueError):
    """The protected schema inventory cannot safely produce a review packet."""


@dataclasses.dataclass(frozen=True)
class SourceReviewPacketBuildResult:
    """Non-sensitive identities of one protected source-review packet."""

    output_root: Path
    packet_path: Path
    packet_sha256: str
    partial_manifest_count: int
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


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SourceReviewPacketBuildError(
                f"schema inventory has duplicate key {key!r}"
            )
        result[key] = value
    return result


def _read_small_regular_file(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        if not path.is_absolute() or path.is_symlink():
            raise SourceReviewPacketBuildError(
                "schema inventory must be an absolute, non-symlink file"
            )
        try:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except OSError as exc:
            raise SourceReviewPacketBuildError(
                "schema inventory cannot be opened safely"
            ) from exc
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_INVENTORY_BYTES:
            raise SourceReviewPacketBuildError(
                "schema inventory must be a small regular file"
            )
        chunks: list[bytes] = []
        total = 0
        while block := os.read(descriptor, 64 * 1024):
            total += len(block)
            if total > _MAX_INVENTORY_BYTES:
                raise SourceReviewPacketBuildError(
                    "schema inventory exceeds the size limit while reading"
                )
            chunks.append(block)
        return b"".join(chunks)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _load_pinned_inventory(path: Path | str) -> tuple[dict[str, Any], str]:
    raw = _read_small_regular_file(Path(path))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != FULL0717_SCHEMA_INVENTORY_SHA256:
        raise SourceReviewPacketBuildError(
            "schema inventory digest does not bind the reviewed full0717 snapshot"
        )
    try:
        decoded = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceReviewPacketBuildError(
            "schema inventory is not valid JSON"
        ) from exc
    if not isinstance(decoded, dict) or raw != _canonical_json(decoded):
        raise SourceReviewPacketBuildError("schema inventory must be canonical JSON")
    historical = decoded.get("historical_export")
    if (
        decoded.get("schema_version") != _SCHEMA_INVENTORY_SCHEMA
        or decoded.get("inventory_ref") != _SCHEMA_INVENTORY_REF
        or decoded.get("metadata_only") is not True
        or decoded.get("source_attested") is not False
        or decoded.get("real_run_authorized") is not False
        or not isinstance(historical, dict)
        or historical.get("export_label") != "full6_20260717"
        or historical.get("export_content_sha256") != FULL0717_EXPORT_CONTENT_SHA256
        or historical.get("export_run_manifest_sha256") != FULL0717_RUN_MANIFEST_SHA256
    ):
        raise SourceReviewPacketBuildError(
            "schema inventory does not retain the full0717 non-authorizing boundary"
        )
    return decoded, digest


def _ensure_new_output_root(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink():
        raise SourceReviewPacketBuildError(
            "output root must be an absolute non-symlink path"
        )
    if candidate.exists():
        raise SourceReviewPacketBuildError("output root must not already exist")
    if not candidate.parent.is_dir() or candidate.parent.is_symlink():
        raise SourceReviewPacketBuildError(
            "output root parent must be a real directory"
        )
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
                raise OSError("failed to write protected source review packet")
            total += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)


def _as_records(value: object, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise SourceReviewPacketBuildError(f"schema inventory {label} must be a list")
    return value


def _source_actions(
    inventory: Mapping[str, Any]
) -> tuple[list[dict[str, object]], int]:
    sources = _as_records(inventory.get("sources"), label="sources")
    source_ids = tuple(source.get("source_id") for source in sources)
    if source_ids != _SOURCE_IDS:
        raise SourceReviewPacketBuildError(
            "schema inventory must retain the exact six-source order"
        )
    actions: list[dict[str, object]] = []
    partial_count = 0
    for source in sources:
        modules = _as_records(source.get("modules"), label="source modules")
        if source.get("module_count") != len(modules):
            raise SourceReviewPacketBuildError(
                "source module count does not match records"
            )
        state_counts = Counter(str(module.get("state")) for module in modules)
        declared_empty = sorted(
            str(module.get("module"))
            for module in modules
            if module.get("state") == "declared_empty"
        )
        partial = []
        for module in modules:
            if module.get("manifest_completeness") != "partial_or_invalid":
                continue
            manifest = module.get("manifest")
            if not isinstance(manifest, dict):
                raise SourceReviewPacketBuildError(
                    "partial manifest record lacks identity"
                )
            partial.append(
                {
                    "module": module.get("module"),
                    "manifest_relative_path": manifest.get("relative_path"),
                    "manifest_sha256": manifest.get("sha256"),
                    "missing_fields": list(module.get("manifest_missing_fields") or []),
                    "invalid_fields": list(module.get("manifest_invalid_fields") or []),
                }
            )
        partial_count += len(partial)
        actions.append(
            {
                "source_id": source["source_id"],
                "export_directory": source.get("export_directory"),
                "module_count": len(modules),
                "module_state_counts": dict(sorted(state_counts.items())),
                "declared_empty_modules": declared_empty,
                "partial_or_invalid_manifests": partial,
                "required_owner_actions": [
                    "attest the native source snapshot and relation schema",
                    "produce a typed column inventory bound to this snapshot",
                    "record source-semantic attestation and owner references",
                ],
            }
        )
    return actions, partial_count


def _selected_members(inventory: Mapping[str, Any]) -> dict[str, list[str]]:
    selected: defaultdict[str, list[str]] = defaultdict(list)
    historical = inventory["historical_export"]
    selected[historical["export_run_manifest_sha256"]].append("run_manifest.json")
    for source in _as_records(inventory.get("sources"), label="sources"):
        for module in _as_records(source.get("modules"), label="source modules"):
            manifest = module.get("manifest")
            if not isinstance(manifest, dict):
                raise SourceReviewPacketBuildError("module lacks manifest identity")
            selected[str(manifest.get("sha256"))].append(
                str(manifest.get("relative_path"))
            )
            data_file = module.get("data_file")
            if isinstance(data_file, dict):
                selected[str(data_file.get("sha256"))].append(
                    str(data_file.get("relative_path"))
                )
    return {digest: sorted(paths) for digest, paths in selected.items()}


def _unlisted_actions(inventory: Mapping[str, Any]) -> list[dict[str, object]]:
    unlisted = _as_records(inventory.get("unlisted_members"), label="unlisted_members")
    selected_by_digest = _selected_members(inventory)
    paths_by_digest: defaultdict[str, list[str]] = defaultdict(list)
    for member in unlisted:
        relative_path = member.get("relative_path")
        digest = member.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(digest, str):
            raise SourceReviewPacketBuildError("unlisted member lacks a file identity")
        paths_by_digest[digest].append(relative_path)

    actions: list[dict[str, object]] = []
    for member in sorted(unlisted, key=lambda item: str(item.get("relative_path"))):
        relative_path = str(member["relative_path"])
        digest = str(member["sha256"])
        selected_duplicates = selected_by_digest.get(digest, [])
        unlisted_duplicates = sorted(paths_by_digest[digest])
        if relative_path.endswith(".DS_Store"):
            classification = "operating_system_metadata"
        elif "/.easyicu_spill/" in f"/{relative_path}":
            classification = "execution_spill_unselected"
        elif selected_duplicates:
            classification = "exact_duplicate_of_selected_member"
        elif len(unlisted_duplicates) > 1:
            classification = "duplicate_unlisted_member"
        elif relative_path.endswith("easyicu_export_manifest.json"):
            classification = "legacy_export_manifest_unselected"
        else:
            classification = "unclassified_unlisted_member"
        actions.append(
            {
                "relative_path": relative_path,
                "sha256": digest,
                "size_bytes": member.get("size_bytes"),
                "classification": classification,
                "exact_duplicate_of_selected_paths": selected_duplicates,
                "exact_duplicate_unlisted_paths": unlisted_duplicates,
                "required_owner_action": "record an explicit include, exclude, or quarantine disposition",
            }
        )
    return actions


def build_source_review_packet(
    *, schema_inventory_path: Path | str, output_root: Path | str
) -> SourceReviewPacketBuildResult:
    """Build one private owner checklist from the exact non-authorizing inventory."""

    inventory, inventory_sha256 = _load_pinned_inventory(schema_inventory_path)
    requested_output = _ensure_new_output_root(output_root)
    source_actions, partial_count = _source_actions(inventory)
    unlisted_actions = _unlisted_actions(inventory)
    if inventory.get("requires_unlisted_member_disposition") != bool(unlisted_actions):
        raise SourceReviewPacketBuildError(
            "schema inventory unlisted-member gate disagrees with its records"
        )
    packet = {
        "schema_version": SOURCE_REVIEW_PACKET_SCHEMA,
        "review_packet_ref": SOURCE_REVIEW_PACKET_REF,
        "schema_inventory_sha256": inventory_sha256,
        "historical_export": inventory["historical_export"],
        "metadata_only": True,
        "source_attested": False,
        "real_run_authorized": False,
        "p4_integration": "forbidden_pending_separate_review",
        "source_review_actions": source_actions,
        "unlisted_member_actions": unlisted_actions,
        "required_signoff": [
            "data owner: exact source snapshot and unlisted-member disposition",
            "transformation owner: typed column inventory and transform semantics",
            "identity owner: native relation and stay-key semantics",
            "methods/clinical reviewer: case-specific typed materialization suitability",
        ],
        "blockers": [
            "SOURCE_DATA_IDENTITY_ATTESTATION_REQUIRED",
            "TYPED_COLUMN_INVENTORY_REQUIRED",
            "UNLISTED_MEMBER_DISPOSITION_REQUIRED",
            "NATIVE_TYPED_MATERIALIZATION_REQUIRED",
            "P4_PRODUCTION_INPUT_AUTHORITY_REQUIRED",
            "FINAL_OPERATOR_FREEZE_REQUIRED",
        ],
    }
    temporary = (
        requested_output.parent / f".{requested_output.name}.tmp-{uuid.uuid4().hex}"
    )
    temporary.mkdir(mode=0o700)
    try:
        packet_path = temporary / "source_review_packet.json"
        packet_raw = _canonical_json(packet)
        _write_private_bytes(packet_path, packet_raw)
        packet_sha256 = hashlib.sha256(packet_raw).hexdigest()
        _write_private_bytes(
            temporary / "build_receipt.json",
            _canonical_json(
                {
                    "schema_version": "easyicu.figure2_source_review_packet_receipt/1",
                    "packet_file": packet_path.name,
                    "packet_sha256": packet_sha256,
                    "schema_inventory_sha256": inventory_sha256,
                    "metadata_only": True,
                    "source_attested": False,
                    "real_run_authorized": False,
                    "partial_manifest_count": partial_count,
                    "unlisted_member_count": len(unlisted_actions),
                }
            ),
        )
        os.replace(temporary, requested_output)
        return SourceReviewPacketBuildResult(
            output_root=requested_output,
            packet_path=requested_output / packet_path.name,
            packet_sha256=packet_sha256,
            partial_manifest_count=partial_count,
            unlisted_member_count=len(unlisted_actions),
        )
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-inventory", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_source_review_packet(
        schema_inventory_path=args.schema_inventory, output_root=args.output_root
    )
    print(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "packet_path": str(result.packet_path),
                "packet_sha256": result.packet_sha256,
                "partial_manifest_count": result.partial_manifest_count,
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
