from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path

import pytest

from easyicu.webserver import cohort_review
from easyicu.webserver.patient_drilldown import coverage
from easyicu.webserver.source_cache_identity import source_file_identity


@pytest.mark.parametrize("size", [4096, 256_000])
@pytest.mark.parametrize("location", ["head", "middle", "tail"])
def test_content_fingerprint_detects_edits_with_identical_stat_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: int, location: str,
) -> None:
    source = tmp_path / "synthetic.bin"
    source.write_bytes(b"a" * size)
    offset = {"head": 0, "middle": size // 2, "tail": size - 1}[location]
    original = source.stat()
    # Model a filesystem on which even ctime does not advance between writes.
    monkeypatch.setattr(Path, "stat", lambda self: original)
    before = source_file_identity(source)
    with source.open("r+b") as handle:
        handle.seek(offset)
        handle.write(b"b")
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))

    assert source_file_identity(source) != before


def test_large_file_identity_reads_bounded_content_on_every_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "synthetic.bin"
    with source.open("wb") as handle:
        handle.truncate(1024 * 1024)
    real_open = Path.open
    bytes_read: list[int] = []

    class Reader:
        def __init__(self, handle):
            self.handle = handle

        def seek(self, offset):
            return self.handle.seek(offset)

        def read(self, size=-1):
            data = self.handle.read(size)
            bytes_read.append(len(data))
            return data

    @contextmanager
    def tracked_open(path, *args, **kwargs):
        with real_open(path, *args, **kwargs) as handle:
            yield Reader(handle)

    monkeypatch.setattr(Path, "open", tracked_open)
    before = source_file_identity(source)
    first_read = sum(bytes_read)
    assert 0 < first_read <= 192 * 1024
    bytes_read.clear()

    assert source_file_identity(source) == before
    assert sum(bytes_read) == first_read


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        ("coverage", "_manifest.json"),
        ("coverage", "outcome.csv"),
        ("summary", "_manifest.json"),
        ("summary", "easyicu_export_manifest.json"),
        ("summary", "outcome.csv"),
    ],
)
def test_review_cache_keys_fingerprint_sources_and_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, owner: str, name: str,
) -> None:
    source = tmp_path / name
    source.write_bytes(b"old-content")
    original = source.stat()
    description = {"files": [{"file": "outcome.csv", "module": "outcome"}]}

    def identity():
        if owner == "coverage":
            return coverage._source_signature(tmp_path, description["files"])
        return cohort_review._summary_cache_key(tmp_path, description, None)

    # Pin only the source's stat evidence, leaving missing-file handling real.
    real_stat = Path.stat
    monkeypatch.setattr(
        Path, "stat", lambda self, *args, **kwargs: (
            original if self == source else real_stat(self, *args, **kwargs)
        ),
    )
    before = identity()
    source.write_bytes(b"new-content")
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))

    assert identity() != before


def test_missing_source_does_not_break_cache_key_building(tmp_path: Path) -> None:
    description = {"files": [{"file": "outcome.csv", "module": "outcome"}]}
    before = cohort_review._summary_cache_key(tmp_path, description, None)
    assert before == cohort_review._summary_cache_key(tmp_path, description, None)
    assert coverage._source_signature(tmp_path, description["files"])
