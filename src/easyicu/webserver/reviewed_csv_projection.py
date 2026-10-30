"""Host-owned, fixed CSV projection for an exactly reviewed source and scope.

This capability prepares one table, not an ICU cohort or an analysis. The
operator performs the actual read in the host and records its complete stream
digest. The source registry and an independently issued, single-use host
review are prerequisites; this module never creates either one.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Mapping, Any

from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.authority.raw_preparation import RawPreparationCheckpoint
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import sources
from easyicu.webserver.raw_case_snapshot import stage_reviewed_source_files, verify_staged_source_files
from easyicu.webserver.raw_preparation_attestation import implementation_identity, publish_attestation, verify_attestation
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding, verify_host_review, consume_host_review


class _ProjectionImplementation:
    pass


def _validated_scope(scope: Mapping[str, Any]) -> tuple[dict, Path]:
    checked = json.loads(canonical_json(dict(scope)))
    if (set(checked) != {"adapter_id", "source_identity", "relative_file", "columns", "purpose", "plan_sha256", "dependencies_sha256"}
            or checked["adapter_id"] != "registered-csv-projection/1" or checked["purpose"] != "prepare_single_table_only"
            or any(not isinstance(checked[name], str) or not re.fullmatch(r"[0-9a-f]{64}", checked[name])
                   for name in ("plan_sha256", "dependencies_sha256"))):
        raise ValueError("csv_projection_scope_invalid")
    identity = SourceIdentity.from_record(checked["source_identity"])
    if identity is None or identity.to_dict() != checked["source_identity"]:
        raise ValueError("csv_projection_source_invalid")
    relative = checked["relative_file"]
    path = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath()
    if (not isinstance(relative, str) or not relative or path.is_absolute() or path.as_posix() != relative
            or any(part in {".", ".."} for part in path.parts) or "\\" in relative or "\x00" in relative or path.suffix.lower() != ".csv"):
        raise ValueError("csv_projection_file_invalid")
    columns = checked["columns"]
    if (not isinstance(columns, list) or not columns or len(columns) > 512
            or any(not isinstance(col, str) or not col or len(col) > 256 for col in columns)
            or len(columns) != len(set(columns))):
        raise ValueError("csv_projection_columns_invalid")
    matches = [Path(row["path"]) for row in sources.load_registry().get("sources", [])
               if row.get("ok") and row.get("path")
               and sources.registered_source_identity(str(row["path"])) == identity]
    if len(matches) != 1 or not matches[0].is_absolute():
        raise ValueError("csv_projection_source_registration_not_unique")
    return checked, matches[0]


def require_reviewable_csv_scope(scope: Mapping[str, Any]) -> None:
    """Validate exact registration and fixed projection before offering review."""
    _validated_scope(scope)


class _DigestingRaw(io.RawIOBase):
    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256()
        self.count = 0

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.handle.read(len(buffer))
        self.digest.update(data)
        self.count += len(data)
        buffer[:len(data)] = data
        return len(data)


def run_reviewed_csv_projection(*, checkpoint: RawPreparationCheckpoint,
                                binding: RawPreparationHostBinding,
                                scope: Mapping[str, Any], host_work_root: Path) -> Path:
    """Run only after the current host has signed this exact scope and plan."""
    checked, source = _validated_scope(scope)
    if checkpoint.plan_sha256 != checked["plan_sha256"]:
        raise ValueError("csv_projection_plan_mismatch")
    coordinates = binding.model_dump(mode="python")
    verify_host_review(checkpoint=checkpoint, scope=checked, **coordinates)
    implementation = implementation_identity(_ProjectionImplementation(), Path(__file__))
    consume_host_review(checkpoint=checkpoint, scope=checked, **coordinates)
    if _validated_scope(checked)[1] != source:
        raise ValueError("csv_projection_registration_changed")
    root = Path(tempfile.mkdtemp(prefix="reviewed-csv-projection-", dir=host_work_root))
    output = root / "output"
    output.mkdir(mode=0o700)
    receipt = stage_reviewed_source_files(source_root=source, staging_root=root / "snapshot",
        relative_files=(checked["relative_file"],),
        source_identity_sha256=canonical_sha256(checked["source_identity"]))
    relative = PurePosixPath(checked["relative_file"])
    parent = root / "snapshot" / Path(*relative.parts[:-1])
    with AnchoredDirectory.open(parent) as directory:
        with directory.open_regular(relative.name) as src, (output / "projection.csv").open("x", encoding="utf-8", newline="") as dst:
            stream = _DigestingRaw(src)
            with io.TextIOWrapper(io.BufferedReader(stream), encoding="utf-8-sig", newline="") as text:
                reader = csv.reader(text, strict=True)
                header = next(reader, None)
                if not header or len(header) != len(set(header)):
                    raise ValueError("csv_projection_header_invalid")
                if any(name not in header for name in checked["columns"]):
                    raise ValueError("csv_projection_column_missing")
                indexes = [header.index(name) for name in checked["columns"]]
                writer = csv.writer(dst, lineterminator="\n")
                writer.writerow(checked["columns"])
                count = 0
                for row in reader:
                    if len(row) != len(header):
                        raise ValueError("csv_projection_row_width_invalid")
                    writer.writerow([row[index] for index in indexes])
                    count += 1
            observed = {"relative_path": checked["relative_file"], "sha256": stream.digest.hexdigest(),
                        "size_bytes": stream.count, "rows": count, "columns": checked["columns"],
                        "method": "host_streamed_csv_projection",
                        "lineage": {"output": "projection.csv",
                                    "rows": "same_one_based_data_record_ordinal",
                                    "columns": "same_named_input_column"}}
        directory.assert_still_selected()
    if (observed["sha256"], observed["size_bytes"]) != (receipt.files[0].sha256, receipt.files[0].size_bytes):
        raise ValueError("csv_projection_consumed_input_changed")
    verify_staged_source_files(root / "snapshot", receipt)
    if implementation_identity(_ProjectionImplementation(), Path(__file__)) != implementation:
        raise ValueError("csv_projection_implementation_changed")
    publish_attestation(root=root, snapshot=receipt, scope=checked, checkpoint_sha256=checkpoint.checkpoint_sha256,
        binding=binding, implementation=implementation, host_consumption=observed)
    verify_attestation(root=root, scope=checked, checkpoint_sha256=checkpoint.checkpoint_sha256,
        binding=binding, adapter=_ProjectionImplementation(), worker=Path(__file__))
    # Private delivery coordinates are verified against the signed receipt and
    # the consumed host decision on every read. They are never sent to clients.
    with AnchoredDirectory.open(root) as directory:
        name, fd = directory.create_temporary(stem="delivery-binding")
        try:
            with os.fdopen(fd, "wb") as stream:
                fd = -1
                stream.write(canonical_json(binding.model_dump(mode="json")).encode())
                stream.flush()
                os.fsync(stream.fileno())
            directory.publish_temporary_no_clobber(name, "delivery-binding.json")
            directory.assert_still_selected()
        finally:
            if fd >= 0:
                os.close(fd)
            directory.unlink(name, missing_ok=True)
    return output


def verify_reviewed_csv_projection(*, root: Path, checkpoint_sha256: str,
                                   binding: RawPreparationHostBinding, scope: Mapping[str, Any]) -> dict:
    checked, _ = _validated_scope(scope)
    return verify_attestation(root=root, scope=checked, checkpoint_sha256=checkpoint_sha256,
        binding=binding, adapter=_ProjectionImplementation(), worker=Path(__file__))


def read_verified_projection(*, root: Path, checkpoint_sha256: str,
                             binding: RawPreparationHostBinding, scope: Mapping[str, Any]) -> bytes:
    """Read exactly the attested projection, with a bounded response size."""
    body = verify_reviewed_csv_projection(root=root, checkpoint_sha256=checkpoint_sha256,
                                          binding=binding, scope=scope)
    files = body.get("outputs")
    if not isinstance(files, list) or len(files) != 1 or files[0].get("path") != "projection.csv":
        raise ValueError("csv_projection_delivery_roster_invalid")
    expected = files[0]
    if expected.get("kind") != "file" or not isinstance(expected.get("size_bytes"), int) or expected["size_bytes"] > 64 * 1024 * 1024:
        raise ValueError("csv_projection_delivery_size_invalid")
    with AnchoredDirectory.open(Path(root) / "output") as directory:
        data = directory.read_bytes("projection.csv", max_bytes=64 * 1024 * 1024)
        directory.assert_still_selected()
    if len(data) != expected["size_bytes"] or hashlib.sha256(data).hexdigest() != expected.get("sha256"):
        raise ValueError("csv_projection_delivery_bytes_changed")
    return data
