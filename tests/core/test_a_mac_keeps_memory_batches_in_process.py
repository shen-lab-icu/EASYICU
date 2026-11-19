"""On macOS the memory rule keeps patient batches in-process.

v0.1.3 RC (S1458): web exports lost their in-process default (9df163b03), so
a low-memory batched export forked a worker. On macOS the child inherited
DuckDB without its scheduler threads and spun forever, with no error. The
rule now keeps those batches in-process on macOS; Linux and Windows keep
process isolation, and an explicit override still wins everywhere.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from easyicu.api import concepts

GIB = 1024**3


@pytest.fixture
def total_ram(monkeypatch):
    import psutil

    def set_total(gib):
        monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=gib * GIB))

    monkeypatch.delenv("EASYICU_FORCE_INPROCESS_BATCH", raising=False)
    monkeypatch.delenv("EASYICU_FORCE_SUBPROCESS_BATCH", raising=False)
    return set_total


@pytest.mark.parametrize("gib,patients", [(8, 5_000), (16, 73_000), (64, 200_000)])
def test_macos_keeps_batches_in_process_where_linux_isolates_them(total_ram, gib, patients):
    total_ram(gib)
    assert concepts._memory_policy_isolates_batches(patients, platform="linux")
    assert concepts._memory_policy_isolates_batches(patients, platform="win32")
    assert not concepts._memory_policy_isolates_batches(patients, platform="darwin")


def test_the_linux_rule_is_unchanged(total_ram):
    total_ram(16)
    assert not concepts._memory_policy_isolates_batches(60_000, platform="linux")
    assert not concepts._memory_policy_isolates_batches(None, platform="linux")
    total_ram(64)
    assert not concepts._memory_policy_isolates_batches(120_000, platform="linux")


def test_an_explicit_override_wins_on_every_platform(total_ram, monkeypatch):
    total_ram(8)
    monkeypatch.setenv("EASYICU_FORCE_SUBPROCESS_BATCH", "1")
    assert concepts._memory_policy_isolates_batches(10, platform="darwin")
    monkeypatch.setenv("EASYICU_FORCE_INPROCESS_BATCH", "1")
    assert not concepts._memory_policy_isolates_batches(10, platform="linux")
