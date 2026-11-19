"""File hashes for synthetic figure presentation verification."""

import hashlib
from collections.abc import Iterable
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_hashes(directory: Path) -> dict[str, str]:
    return {path.name: sha(path) for path in directory.glob("*.csv")}


def export_hashes(directory: Path, deliveries: Iterable[str]) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha(path)
        for delivery in deliveries
        for path in (directory / delivery).iterdir()
        if path.suffix in {".svg", ".pdf", ".png"}
    }
