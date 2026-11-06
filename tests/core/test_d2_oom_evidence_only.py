"""Out-of-memory decisions use typed evidence only (lead-r13 B-RETRY, ruling C).

Codex delta2: verify that ruling C replaced the old classifier, which counted
exit statuses (-9/9/137/Windows codes) and memory words in a traceback or a
module error as out-of-memory. The only exit statuses still read are ruling
C's bare-kill set, and only for the one unverified retry on a host without an
OOM counter. The native-export driver followed the same old rule and now
reads the package's typed classification.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import easyicu.api.extraction as extraction

REPO = Path(__file__).resolve().parents[2]
_FILES = [
    REPO / "src/easyicu/api/extraction.py",
    REPO / "tools/reextract_native_export_v2.py",
]
_MARKERS = ("memoryerror", "out of memory", "outofmemory", "cannot allocate", "bad_alloc",
            "exhausted memory", "oom", "killed")


def _membership_strings(tree):
    """String constants used as the needle or haystack of an ``in`` test."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            for side in [node.left, *node.comparators]:
                for leaf in ast.walk(side):
                    if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str):
                        yield leaf
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "any":
            for leaf in ast.walk(node):
                if isinstance(leaf, ast.Constant) and isinstance(leaf.value, str):
                    yield leaf


@pytest.mark.parametrize("path", _FILES, ids=lambda p: p.name)
def test_no_memory_decision_searches_text(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = [
        f"{path.name}:{leaf.lineno} {leaf.value!r}"
        for leaf in _membership_strings(tree)
        if any(marker in leaf.value.lower() for marker in _MARKERS)
    ]
    assert hits == []


def test_exit_statuses_are_read_only_by_the_bare_kill_rule():
    tree = ast.parse(_FILES[0].read_text(encoding="utf-8"))
    readers = set()
    for function in ast.walk(tree):
        if isinstance(function, ast.FunctionDef):
            if any(isinstance(n, ast.Name) and n.id == "_UNVERIFIED_KILL_EXIT_CODES" for n in ast.walk(function)):
                readers.add(function.name)
    assert readers == {"_unverified_kill_retry_eligible", "_worker_failure_class"}
    for name in ("_MEMORY_EXIT_CODES", "_MEMORY_FAILURE_MARKERS", "_worker_failure_is_memory"):
        assert not hasattr(extraction, name)


@pytest.mark.parametrize("code", [-9, 9, 137, -1073740791, 3221226505, 0xC0000017])
def test_an_exit_status_or_traceback_text_is_not_memory_evidence(code):
    record = {"worker_exit_code": code, "traceback": "MemoryError: out of memory (std::bad_alloc)",
              "failure_kind": "worker_exit_without_failure_record"}
    assert extraction._worker_memory_evidence(code, [record]) is None
    assert extraction.worker_failure_is_memory(record) is False
    assert extraction._worker_failure_class(code, [record]) != "out_of_memory"


def test_typed_evidence_is_memory():
    assert extraction._worker_memory_evidence(-9, [], oom_kill_delta=1) == "oom_kill_event"
    assert extraction._worker_memory_evidence(1, [{"memory_exhausted": True}]) == "memory_exception"
    assert extraction._worker_memory_evidence(1, [{"exception_type": "MemoryError"}]) == "memory_exception"
    assert extraction.is_memory_exception(MemoryError())
    assert not extraction.is_memory_exception(ValueError("out of memory"))


@pytest.mark.parametrize("has_counter", [False, True])
def test_bare_kill_retry_only_without_an_oom_counter(monkeypatch, has_counter):
    monkeypatch.setattr(extraction, "_host_reports_oom_kills", lambda: has_counter)
    for code in (-9, 137, 0xC0000017):
        assert extraction._unverified_kill_retry_eligible(code, []) is (not has_counter)
    # Outside the declared bare-kill set, or after a recorded exception: never.
    assert extraction._unverified_kill_retry_eligible(9, []) is False
    assert extraction._unverified_kill_retry_eligible(0xC0000409, []) is False
    assert extraction._unverified_kill_retry_eligible(-9, [{"exception_type": "ValueError"}]) is False


def test_module_memory_errors_read_the_typed_flag_not_error_text(tmp_path):
    for module, manifest in {
        "texty": {"errors": ["streamed module export exhausted memory: texty (out of memory)"]},
        "typed": {"errors": ["x"], "memory_exhausted": True},
    }.items():
        (tmp_path / module).mkdir()
        (tmp_path / module / "_manifest.json").write_text(json.dumps(manifest))
    assert extraction._module_memory_errors(str(tmp_path), ["texty", "typed"]) == ["typed"]
