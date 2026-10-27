"""Bind a desktop installer artifact to its source commit before acceptance or release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

# Each platform's installer file and the build target its receipt must name.
PLATFORMS = {
    "windows": ("*.exe", "x86_64-pc-windows-msvc"),
    "macos": ("*.dmg", "aarch64-apple-darwin"),
}


def verify_installer(root: Path, source_commit: str, platform: str = "windows") -> tuple[Path, str]:
    pattern, target = PLATFORMS[platform]
    installers = sorted(root.rglob(pattern))
    receipts = sorted(root.rglob("build-inputs.json"))
    if len(installers) != 1 or len(receipts) != 1:
        raise ValueError(f"{platform}_artifact_ambiguous_or_missing")
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    if (receipt.get("source_commit") != source_commit or receipt.get("source_dirty") is not False
            or receipt.get("target") != target):
        raise ValueError(f"{platform}_artifact_source_mismatch")
    installer = installers[0].resolve()
    with installer.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if receipt.get("installer") != {"name": installer.name, "sha256": digest}:
        raise ValueError(f"{platform}_artifact_digest_mismatch")
    return installer, digest


def release_name(name: str) -> str:
    """The asset name GitHub keeps as uploaded.

    GitHub rewrites other characters (a space becomes "."), and SHA256SUMS
    must name the file a downloader actually gets.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", ".", name).strip(".")


def _record_checksum(release_dir: Path, name: str, digest: str) -> None:
    """Add one line to SHA256SUMS.txt, keeping the other platforms' lines."""
    sums = release_dir / "SHA256SUMS.txt"
    entries = {}
    if sums.is_file():
        for line in sums.read_text(encoding="utf-8").splitlines():
            recorded, _, recorded_name = line.partition("  ")
            if recorded_name:
                entries[recorded_name] = recorded
    entries[name] = digest
    sums.write_bytes("".join(f"{entries[key]}  {key}\n" for key in sorted(entries)).encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("platform", choices=sorted(PLATFORMS))
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--release-dir", type=Path)
    args = parser.parse_args(argv)
    installer, digest = verify_installer(args.artifact_dir, args.source_commit, args.platform)
    if args.release_dir is not None:
        args.release_dir.mkdir(parents=True, exist_ok=True)
        target = args.release_dir / release_name(installer.name)
        shutil.copy2(installer, target)
        _record_checksum(args.release_dir, target.name, digest)
        installer = target.resolve()
    print(installer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
