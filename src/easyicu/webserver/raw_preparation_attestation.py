"""Host-signed identity of one reviewed producer's staged inputs and outputs.

This receipt is not proof of an atomic source snapshot, a complete dependency
closure, or actual worker reads. Same-UID processes can access host state; this
mechanism is not an isolation boundary against that user.
"""
from __future__ import annotations

import hashlib
import hmac
import inspect
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Mapping

from easyicu import state_paths
from easyicu.research_agent.authority.filesystem import AnchoredDirectory
from easyicu.research_agent.canonical_json import canonical_json, canonical_sha256
from easyicu.webserver.raw_case_snapshot import SnapshotFile, StagedRawSnapshot, verify_staged_source_files
from easyicu.webserver.raw_preparation_decisions import RawPreparationHostBinding

_SCHEMA = "easyicu.reviewed-snapshot-output/1"


def _hash_file(path: Path) -> str:
    path = Path(path)
    with AnchoredDirectory.open(path.parent) as directory:
        with directory.open_regular(path.name) as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        directory.assert_still_selected()
    return digest


def implementation_identity(adapter: Any, worker: Path) -> dict[str, str]:
    """Record the inspected adapter module and worker, not the entire runtime."""
    filename = inspect.getsourcefile(type(adapter))
    if not filename:
        raise ValueError("raw_output_adapter_source_unavailable")
    return {"adapter_sha256": _hash_file(Path(filename).absolute()),
            "worker_sha256": _hash_file(Path(worker).absolute())}


def _key(*, create: bool) -> bytes:
    root = state_paths.state_root() / "private-preparation-attestations"
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    with AnchoredDirectory.open(root) as directory:
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise ValueError("raw_output_attestation_directory_invalid")
        if create:
            name, fd = directory.create_temporary(stem="host-key")
            try:
                with os.fdopen(fd, "wb") as handle:
                    fd = -1
                    handle.write(secrets.token_bytes(32))
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    directory.publish_temporary_no_clobber(name, "host.key")
                except FileExistsError:
                    pass
            finally:
                if fd >= 0:
                    os.close(fd)
                directory.unlink(name, missing_ok=True)
        with directory.open_regular("host.key") as handle:
            info = os.fstat(handle.fileno())
            if info.st_mode & 0o077 or info.st_uid != os.getuid():
                raise ValueError("raw_output_attestation_key_invalid")
            value = handle.read(33)
        directory.assert_still_selected()
    if len(value) != 32:
        raise ValueError("raw_output_attestation_key_invalid")
    return value


def _files(root: Path) -> list[dict[str, Any]]:
    """Hash the entire output tree, rejecting symlinks and non-regular files."""
    rows: list[dict[str, Any]] = []
    pending = [Path()]
    while pending:
        relative = pending.pop()
        path = root / relative
        with AnchoredDirectory.open(path) as directory:
            for name in sorted(os.listdir(path)):
                if name in {".", ".."} or "/" in name or "\\" in name:
                    raise ValueError("raw_output_name_invalid")
                info = directory.stat(name)
                child = relative / name
                if len(child.parts) > 16 or len(child.as_posix()) > 1024 or len(rows) >= 10000:
                    raise ValueError("raw_output_roster_too_large")
                if stat.S_ISDIR(info.st_mode):
                    rows.append({"path": child.as_posix(), "kind": "directory"})
                    pending.append(child)
                elif stat.S_ISREG(info.st_mode):
                    with directory.open_regular(name) as handle:
                        file_info = os.fstat(handle.fileno())
                        if file_info.st_nlink != 1:
                            raise ValueError("raw_output_hardlink_not_allowed")
                        digest = hashlib.file_digest(handle, "sha256").hexdigest()
                    rows.append({"path": child.as_posix(), "kind": "file", "sha256": digest,
                                 "size_bytes": file_info.st_size})
                else:
                    raise ValueError("raw_output_not_regular")
            directory.assert_still_selected()
    if not any(item["kind"] == "file" for item in rows):
        raise ValueError("raw_output_empty")
    return sorted(rows, key=lambda item: item["path"])


def publish_attestation(*, root: Path, snapshot: StagedRawSnapshot,
                        scope: Mapping[str, Any], checkpoint_sha256: str,
                        binding: RawPreparationHostBinding, implementation: Mapping[str, str],
                        host_consumption: Mapping[str, Any] | None = None) -> None:
    """Write a private receipt only after the host validated worker outputs."""
    root = Path(root)
    body = {"schema_version": _SCHEMA, "scope_sha256": canonical_sha256(dict(scope)),
            "checkpoint_sha256": checkpoint_sha256,
            "binding_sha256": canonical_sha256(binding.model_dump(mode="json")),
            "snapshot": snapshot.public_receipt(), "implementation": dict(implementation),
            "outputs": _files(root / "output"), "actual_consumption_verified": False,
            "read_roster_complete": False, "point_in_time_source_snapshot_verified": False,
            "runtime_integrity_verified": False, "analysis_authorized": False}
    if host_consumption is not None:
        # Evidence is a host stream observation, not an independent OS trace.
        body["host_consumption"] = dict(host_consumption)
    payload = canonical_json(body).encode()
    signature = hmac.new(_key(create=True), payload, hashlib.sha256).hexdigest()
    data = canonical_json({"body": body, "signature": signature}).encode()
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("raw_output_attestation_too_large")
    with AnchoredDirectory.open(root) as directory:
        name, fd = directory.create_temporary(stem="producer-attestation")
        try:
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            directory.assert_still_selected()
            directory.publish_temporary_no_clobber(name, "producer-attestation.json")
        finally:
            if fd >= 0:
                os.close(fd)
            directory.unlink(name, missing_ok=True)


def verify_attestation(*, root: Path, scope: Mapping[str, Any], checkpoint_sha256: str,
                       binding: RawPreparationHostBinding, adapter: Any, worker: Path) -> dict:
    """Recheck signed coordinates and all staged/output bytes without granting access."""
    root = Path(root)
    with AnchoredDirectory.open(root) as directory:
        raw = directory.read_bytes("producer-attestation.json", max_bytes=4 * 1024 * 1024)
        directory.assert_still_selected()
    try:
        receipt = json.loads(raw)
        body, signature = receipt["body"], receipt["signature"]
        expected = hmac.new(_key(create=False), canonical_json(body).encode(), hashlib.sha256).hexdigest()
        if not isinstance(signature, str) or not hmac.compare_digest(signature, expected):
            raise ValueError("raw_output_attestation_signature_invalid")
        if (body.get("schema_version") != _SCHEMA
                or body.get("scope_sha256") != canonical_sha256(dict(scope))
                or body.get("checkpoint_sha256") != checkpoint_sha256
                or body.get("binding_sha256") != canonical_sha256(binding.model_dump(mode="json"))):
            raise ValueError("raw_output_attestation_binding_mismatch")
        if body.get("implementation") != implementation_identity(adapter, worker):
            raise ValueError("raw_output_implementation_changed")
        stored = body["snapshot"]
        snapshot = StagedRawSnapshot(source_identity_sha256=stored["source_identity_sha256"],
            files=tuple(SnapshotFile(**item) for item in stored["files"]))
        if snapshot.public_receipt() != stored:
            raise ValueError("raw_output_snapshot_receipt_invalid")
        verify_staged_source_files(root / "snapshot", snapshot)
        if _files(root / "output") != body.get("outputs"):
            raise ValueError("raw_output_bytes_changed")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("raw_output_attestation_invalid") from exc
    return body
