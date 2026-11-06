"""The re-extraction driver classifies memory failures by the package's rule.

lead-r13 D4-DRIVER-OOM (Codex delta4): ``tools/reextract_native_export_v2.py``
asks the package (``worker_failure_is_memory``) whether a nested worker ran
out of memory and, if so, retries smaller. The shared cgroup counter could
make a recorded ``ValueError`` a memory failure there too. The package rule
applies: a record's own non-memory exception wins; the counter classifies
only a SIGKILL that left no record; anything else is not memory evidence.
Synthetic records only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from easyicu.api import extraction as api

TOOL = Path(__file__).resolve().parents[2] / "tools" / "reextract_native_export_v2.py"
SPEC = importlib.util.spec_from_file_location("reextract_native_export_v2_d4", TOOL)
assert SPEC and SPEC.loader
driver = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = driver
SPEC.loader.exec_module(driver)

VALUE_ERROR = {"exception_type": "ValueError", "memory_exhausted": False}


@pytest.mark.parametrize("exit_code", [1, -9, 137])
def test_a_recorded_value_error_with_a_rising_counter_is_not_retried(exit_code):
    record = {**VALUE_ERROR, "worker_exit_code": exit_code, "oom_kill_delta": 1}
    assert not api.worker_failure_is_memory(record)
    assert driver._worker_memory_classification(RuntimeError("x"), [record]) == (False, False)


def test_a_stored_memory_class_does_not_override_the_records_own_exception():
    record = {**VALUE_ERROR, "failure_class": "out_of_memory", "worker_exit_code": 1}
    assert not api.worker_failure_is_memory(record)
    assert driver._worker_memory_classification(RuntimeError("x"), [record])[1] is False


@pytest.mark.parametrize("exit_code", [1, -15, 143])
def test_the_counter_alone_does_not_explain_a_non_sigkill_exit(exit_code):
    record = {"exception_type": None, "worker_exit_code": exit_code, "oom_kill_delta": 1}
    assert driver._worker_memory_classification(RuntimeError("x"), [record])[1] is False


def test_attributed_evidence_still_retries():
    unrecorded_kill = {"exception_type": None, "worker_exit_code": -9, "oom_kill_delta": 1}
    assert driver._worker_memory_classification(RuntimeError("x"), [unrecorded_kill])[1] is True
    assert driver._worker_memory_classification(
        RuntimeError("x"), [{"exception_type": "MemoryError", "worker_exit_code": 1}]
    )[1] is True
    assert driver._worker_memory_classification(
        RuntimeError("x"), [{"failure_class": "out_of_memory", "worker_exit_code": -9}]
    )[1] is True


def test_the_driver_uses_the_package_classifier_not_a_copy(monkeypatch):
    seen = []

    def spy(record):
        seen.append(record)
        return False

    monkeypatch.setattr(api, "worker_failure_is_memory", spy)
    record = {**VALUE_ERROR, "worker_exit_code": -9, "oom_kill_delta": 1}
    driver._worker_memory_classification(RuntimeError("x"), [record])
    assert seen == [record]
