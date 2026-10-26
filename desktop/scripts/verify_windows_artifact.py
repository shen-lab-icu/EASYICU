"""Bind an installer artifact to its source commit before acceptance or release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def verify_installer(root: Path, source_commit: str) -> tuple[Path, str]:
    installers = sorted(root.rglob("*.exe"))
    receipts = sorted(root.rglob("build-inputs.json"))
    if len(installers) != 1 or len(receipts) != 1:
        raise ValueError("windows_artifact_ambiguous_or_missing")
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    if (receipt.get("source_commit") != source_commit or receipt.get("source_dirty") is not False
            or receipt.get("target") != "x86_64-pc-windows-msvc"):
        raise ValueError("windows_artifact_source_mismatch")
    installer = installers[0].resolve()
    with installer.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if receipt.get("installer") != {"name": installer.name, "sha256": digest}:
        raise ValueError("windows_artifact_digest_mismatch")
    return installer, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--release-dir", type=Path)
    args = parser.parse_args(argv)
    installer, digest = verify_installer(args.artifact_dir, args.source_commit)
    if args.release_dir is not None:
        args.release_dir.mkdir(parents=True, exist_ok=True)
        target = args.release_dir / installer.name
        shutil.copy2(installer, target)
        (args.release_dir / "SHA256SUMS.txt").write_bytes(f"{digest}  {installer.name}\n".encode("utf-8"))
        installer = target.resolve()
    print(installer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
