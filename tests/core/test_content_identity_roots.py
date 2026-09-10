import json
import os

import pytest

from easyicu import content_identity as identity


def test_shared_index_preserves_roots_and_detects_replaced_content(
    tmp_path, monkeypatch
):
    roots = [tmp_path / name for name in ("a", "b")]
    for root in roots:
        root.mkdir()
        (root / "data.csv").write_text("n\n1\n")
    cache = tmp_path / "cache"
    hashed = []
    real_hash = identity._sha256_file

    def tracked_hash(path):
        hashed.append(path)
        return real_hash(path)

    monkeypatch.setattr(identity, "_sha256_file", tracked_hash)
    results = []
    for root in (roots[0], roots[0], roots[1], roots[0]):
        results.append(identity.data_path_fingerprint(root, exclude_dir=cache))
    assert hashed == [roots[0] / "data.csv", roots[1] / "data.csv"]
    assert results[0] == results[1] == results[3]
    index = cache / identity._CONTENT_RECEIPT_INDEX
    before = index.stat().st_mtime_ns
    identity.data_path_fingerprint(roots[0], exclude_dir=cache)
    assert index.stat().st_mtime_ns == before

    source = roots[0] / "data.csv"
    original = source.stat()
    replacement = roots[0] / "replacement"
    replacement.write_text("n\n2\n")
    os.utime(replacement, ns=(original.st_atime_ns, original.st_mtime_ns))
    os.replace(replacement, source)
    assert identity.data_path_fingerprint(roots[0], exclude_dir=cache) != results[0]
    assert len(json.loads(index.read_text())["roots"]) == 2


@pytest.mark.parametrize("legacy", [True, False])
def test_receipt_index_migration_or_corruption_is_recoverable(
    tmp_path, monkeypatch, legacy
):
    root = tmp_path / "data"
    root.mkdir()
    source = root / "data.csv"
    source.write_text("n\n1\n")
    cache = tmp_path / "cache"
    cache.mkdir()
    index = cache / identity._CONTENT_RECEIPT_INDEX
    receipt = identity.file_content_receipt(source)
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root": str(root),
                "files": {"data.csv": receipt},
            }
        )
        if legacy
        else "[]"
    )
    if legacy:
        monkeypatch.setattr(
            identity,
            "_sha256_file",
            lambda path: pytest.fail("warm legacy receipt was rehashed"),
        )
    fingerprint = identity.data_path_fingerprint(root, exclude_dir=cache)
    assert fingerprint
    payload = json.loads(index.read_text())
    assert payload["schema_version"] == 2
    assert payload["roots"][str(root)]["data.csv"]["sha256"] == receipt["sha256"]
