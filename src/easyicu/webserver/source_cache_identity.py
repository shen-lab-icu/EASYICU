"""Bounded content fingerprints for process-local review caches."""

from __future__ import annotations

import hashlib
from pathlib import Path


_BLOCK_BYTES = 64 * 1024


def source_file_identity(path: Path) -> tuple[int, int, int, int, int, str]:
    """Combine stat evidence with content, even when filesystem clocks collide.

    Small files are hashed completely. Large exports contribute 64 KiB each
    from their head, middle and tail, keeping interactive checks independent
    of export size. Always re-read these blocks: caching the digest by stat
    would reintroduce same-tick, restored-mtime collisions.

    This is a cache fingerprint, not an integrity receipt: an edit outside
    the sampled blocks of a large file still needs changed stat evidence.
    """
    stat = path.stat()
    size = stat.st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if size <= 3 * _BLOCK_BYTES:
            digest.update(handle.read(size))
        else:
            for offset in (0, (size - _BLOCK_BYTES) // 2, size - _BLOCK_BYTES):
                handle.seek(offset)
                digest.update(handle.read(_BLOCK_BYTES))
    return (
        int(size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
        int(stat.st_dev),
        int(stat.st_ino),
        digest.hexdigest(),
    )
