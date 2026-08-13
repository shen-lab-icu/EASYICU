"""Characterization of the immutable-receipt write-once contract.

Six places publish an immutable receipt: reviewed memory, the coder resource
snapshot, the cross-run memory store, and the three capability records. Each one
hand-rolled the same sequence -- reconcile an existing file by bytes, write to a
sibling temp file, fsync, hard-link it into place, re-reconcile on EEXIST, drop
the temp file -- and each carried its own exception type and temp prefix.

These tests are written against the *observable* behaviour of the callers, not
against the shared helper, so they keep meaning after the six copies collapse
into one contract. Every case pins the three answers that matter for an
integrity receipt:

  * republishing identical bytes is a no-op, not an error;
  * republishing different bytes raises that caller's declared exception, and
    the already-published bytes survive;
  * no temp file is left behind, on either path.

The hard link is what makes this write-*once* rather than write-atomically:
``os.replace`` would let a second writer silently overwrite a receipt someone
else had already published, so the loser of a race would never learn it lost.
The remaining callers (capability records, the memory store) are guarded by
their own suites -- ``test_capability_requests.py``,
``test_permissioned_memory_store.py``, ``test_memory.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.learning.runtime import (
    ReviewedMemoryIntegrityError,
    _write_once as _reviewed_memory_write_once,
)
from easyicu.research_agent.resources.coder import (
    CoderResourceIntegrityError,
    _write_once as _coder_resource_write_once,
)


def _temp_leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.startswith("."))


@pytest.mark.parametrize(
    ("write_once", "error"),
    [
        (_reviewed_memory_write_once, ReviewedMemoryIntegrityError),
        (_coder_resource_write_once, CoderResourceIntegrityError),
    ],
    ids=["reviewed_memory", "coder_resource"],
)
def test_a_receipt_is_written_once_and_reconciled_by_bytes(
    tmp_path: Path, write_once, error
) -> None:
    path = tmp_path / "nested" / "receipt.json"

    write_once(path, b"payload")
    assert path.read_bytes() == b"payload"
    assert _temp_leftovers(path.parent) == []

    # Identical bytes: this is what a retry looks like, not a failure.
    write_once(path, b"payload")
    assert path.read_bytes() == b"payload"
    assert _temp_leftovers(path.parent) == []

    with pytest.raises(error):
        write_once(path, b"different")
    # The refusal must not have damaged the receipt that was already published.
    assert path.read_bytes() == b"payload"
    assert _temp_leftovers(path.parent) == []


@pytest.mark.parametrize(
    ("write_once", "error"),
    [
        (_reviewed_memory_write_once, ReviewedMemoryIntegrityError),
        (_coder_resource_write_once, CoderResourceIntegrityError),
    ],
    ids=["reviewed_memory", "coder_resource"],
)
def test_the_loser_of_a_publish_race_is_told(
    tmp_path: Path, monkeypatch, write_once, error
) -> None:
    """A file appearing between the existence check and the link must raise.

    This is the property ``os.replace`` would destroy: the second writer would
    win silently and the first writer's receipt would vanish with no error.
    """

    import os as os_module

    path = tmp_path / "receipt.json"
    real_link = os_module.link
    raced: dict[str, bool] = {}

    def _link(src, dst, **kwargs):
        if not raced:
            raced["yes"] = True
            # Someone else publishes different bytes first.
            Path(dst).write_bytes(b"theirs")
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(os_module, "link", _link)

    with pytest.raises(error):
        write_once(path, b"mine")
    assert raced == {"yes": True}
    assert path.read_bytes() == b"theirs"
    assert _temp_leftovers(tmp_path) == []


def test_a_race_that_publishes_identical_bytes_is_not_an_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Losing a race to the SAME bytes is still a successful publish."""

    import os as os_module

    path = tmp_path / "receipt.json"
    payload = json.dumps({"v": 1}).encode()
    real_link = os_module.link
    raced: dict[str, bool] = {}

    def _link(src, dst, **kwargs):
        if not raced:
            raced["yes"] = True
            Path(dst).write_bytes(payload)
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(os_module, "link", _link)

    _reviewed_memory_write_once(path, payload)
    assert raced == {"yes": True}
    assert path.read_bytes() == payload
    assert _temp_leftovers(tmp_path) == []
