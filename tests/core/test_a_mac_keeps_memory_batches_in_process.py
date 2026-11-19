"""On macOS the memory rule keeps patient batches in-process.

v0.1.3 RC (S1458): web exports lost their in-process default (9df163b03), so
a low-memory batched export forked a worker. On macOS the child inherited
DuckDB without its scheduler threads and spun forever, with no error. The
RAM rule and the one-shot budget now keep those batches in-process on macOS;
Linux and Windows keep process isolation, and an explicit override still
wins everywhere. Special concepts still need workers (S1459 known issue).
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu import api
from easyicu.api import concepts

GIB = 1024**3


@pytest.fixture
def on(monkeypatch):
    import psutil

    def set_platform(name, gib=8):
        monkeypatch.setattr(concepts, "_platform_keeps_batches_in_process", lambda: name == "darwin")
        monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=gib * GIB, available=gib * GIB))

    for key in ("EASYICU_FORCE_INPROCESS_BATCH", "EASYICU_FORCE_SUBPROCESS_BATCH", "EASYICU_ONESHOT_BUDGET_MB",
                "EASYICU_AUTO_CHUNK_SIZE", "EASYICU_DISABLE_AUTO_CHUNK"):
        monkeypatch.delenv(key, raising=False)
    return set_platform


def test_load_concepts_keeps_its_resolver_scope():
    # Carbon b8: the helper once sat between the decorator and load_concepts.
    assert concepts.load_concepts.__wrapped__.__name__ == "load_concepts"
    assert not hasattr(concepts._memory_policy_isolates_batches, "__wrapped__")


@pytest.mark.parametrize("gib,patients", [(8, 5_000), (16, 73_000), (64, 200_000)])
def test_macos_keeps_batches_in_process_where_linux_isolates_them(on, gib, patients):
    on("linux", gib)
    assert concepts._memory_policy_isolates_batches(patients)
    on("darwin", gib)
    assert not concepts._memory_policy_isolates_batches(patients)


def test_the_linux_rule_is_unchanged(on):
    on("linux", 16)
    assert not concepts._memory_policy_isolates_batches(60_000)
    assert not concepts._memory_policy_isolates_batches(None)
    on("linux", 64)
    assert not concepts._memory_policy_isolates_batches(120_000)


def test_an_explicit_override_wins_on_every_platform(on, monkeypatch):
    on("darwin")
    monkeypatch.setenv("EASYICU_FORCE_SUBPROCESS_BATCH", "1")
    assert concepts._memory_policy_isolates_batches(10)
    on("linux")
    monkeypatch.setenv("EASYICU_FORCE_INPROCESS_BATCH", "1")
    assert not concepts._memory_policy_isolates_batches(10)


@pytest.mark.parametrize("site", ["named-stays", "all-stays"])
@pytest.mark.parametrize("platform,route", [("linux", "subprocess"), ("darwin", "in-process")])
def test_a_module_over_the_one_shot_budget_batches_in_process_on_macos(tmp_path, monkeypatch, on, platform, route, site):
    from easyicu.runtime import memory_manager

    on(platform, 64)  # the RAM rule alone would not isolate 5,001 stays
    monkeypatch.setenv("EASYICU_DISABLE_AUTO_CHUNK", "1")
    ids = list(range(1, 5_002))
    calls = []

    def record(name):
        def load(**kwargs):
            calls.append(name)
            return pd.DataFrame({"stay_id": [1], "hr": [61.0]})
        return load

    loader = SimpleNamespace(database="miiv", data_path=tmp_path, load_concepts=record("one-shot"))
    monkeypatch.setattr(concepts, "_get_global_loader", lambda **kwargs: loader)
    monkeypatch.setattr(concepts, "_get_total_patient_count", lambda _loader: len(ids))
    monkeypatch.setattr(concepts, "_sample_patient_ids", lambda *args, **kwargs: ids)
    monkeypatch.setattr(memory_manager, "estimate_memory_mb", lambda *args: 10**9)
    monkeypatch.setattr(memory_manager, "auto_batch_size", lambda *args, **kwargs: 2_501)
    monkeypatch.setattr(memory_manager, "inprocess_batch_load", record("in-process"))
    monkeypatch.setattr(memory_manager, "subprocess_batch_load", record("subprocess"))
    api.load_concepts(["hr"], database="miiv", data_path=tmp_path,
                      patient_ids={"stay_id": ids} if site == "named-stays" else None,
                      concept_workers=1, parallel_workers=1, merge=True, r_compatible=False, verbose=False)
    assert calls == [route]
