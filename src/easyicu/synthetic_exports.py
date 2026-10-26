"""App-owned receipts for exports produced from verified generated sources.

Receipts live in instance state, never in the caller's source/export folder.
Only the native web producer records them, after it verifies its input before
and after extraction. Export labels and caller-supplied hashes grant nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from easyicu.state_paths import state_root
from easyicu.synthetic_sources import verify_source


def _receipt_path(export: Path) -> Path:
    key = hashlib.sha256(str(export).encode()).hexdigest()
    return state_root() / "synthetic_export_receipts" / f"{key}.json"


def _inventory(root: Path) -> dict[str, str]:
    if root.resolve() != root:
        raise ValueError("Synthetic exports cannot traverse symlinks")
    result = {}
    for member in sorted(root.rglob("*")):
        if member.is_symlink():
            raise ValueError("Synthetic exports cannot contain symlinks")
        if member.is_file():
            with member.open("rb") as stream:
                result[member.relative_to(root).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def record_generated_export(source: str | Path, export: str | Path, *, source_proof: dict | None) -> None:
    """Producer-only terminal write; no public API accepts a synthetic claim."""
    if source_proof is None:
        return
    source_path, root = Path(source).absolute(), Path(export).absolute()
    if verify_source(source_path) != source_proof:
        raise ValueError("Synthetic input changed during extraction")
    manifest = json.loads((root / "_manifest.json").read_text(encoding="utf-8"))
    from easyicu.databases.profiles import get_database_profile
    if (get_database_profile(manifest["database"]).key != source_proof["database"]
            or manifest.get("source_kind") != "synthetic"):
        raise ValueError("Synthetic export identity does not match its input")
    expected = {"_manifest.json", "README.md", manifest["column_metadata"]["file"]}
    expected.update(row["file"] for row in manifest["files"])
    expected.update(row["file"] for row in manifest.get("feature_definitions", {}).get("files", []))
    inventory = _inventory(root)
    if set(inventory) != expected:
        raise ValueError("Synthetic export contains files not written by the producer")
    receipt = {"schema_version": "easyicu.synthetic_export_receipt/1", "export": str(root),
               "source": str(source_path), "source_proof": source_proof, "files": inventory}
    target = _receipt_path(root)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = target.with_name(f".{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(receipt, stream, sort_keys=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def verified_export_source(export: str | Path) -> dict | None:
    """Recheck instance-owned lineage and every byte before granting clearance."""
    root = Path(export).absolute()
    target = _receipt_path(root)
    try:
        if target.resolve() != target or not target.is_file() or target.stat().st_size > 256 * 1024:
            return None
        receipt = json.loads(target.read_text(encoding="utf-8"))
        if (receipt.get("schema_version") != "easyicu.synthetic_export_receipt/1"
                or receipt.get("export") != str(root) or receipt.get("files") != _inventory(root)):
            return None
        proof = verify_source(receipt["source"])
        return proof if proof is not None and proof == receipt.get("source_proof") else None
    except (OSError, ValueError, TypeError, KeyError):
        return None
