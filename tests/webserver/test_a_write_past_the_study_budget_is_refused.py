"""A write past a study's budget is refused, and the store stays readable.

Every read holds each stored study to the context budget, so a single study
past it would leave the whole store unreadable.  A patch within the budget
can still merge into a study past it.  That write is refused with a typed
reason, and every study -- the one written included -- reads as it did.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.webserver import study_contexts as context_store

_STUDY = "study_budget00001"
_OTHER = "study_budget00002"


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        context_store, "_CONFIG_PATH", tmp_path / "cfg" / "study-contexts.json"
    )


def _bytes(row: dict) -> int:
    return len(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode())


def _modules(count: int) -> dict:
    return {"modules": [f"module_{index}" for index in range(count)]}


def _text(count: int) -> dict:
    return {"purpose": "x" * count}


@pytest.mark.parametrize(
    ("limit", "measure", "headroom", "growth", "small", "large", "code"),
    [
        # Each patch is a few dozen nodes, well within the budget alone.
        (
            "_MAX_CONTEXT_NODES",
            context_store._metadata_node_count,
            10,
            _modules,
            8,
            16,
            "study_context_row_too_complex",
        ),
        ("_MAX_CONTEXT_BYTES", _bytes, 600, _text, 400, 800, "study_context_row_too_large"),
    ],
    ids=["nodes", "bytes"],
)
def test_a_patch_that_merges_past_the_budget_is_refused(
    monkeypatch: pytest.MonkeyPatch, limit, measure, headroom, growth, small, large, code
) -> None:
    other = context_store.upsert_context({"id": _OTHER, "question": "q"})
    created = context_store.upsert_context({"id": _STUDY, "question": "q"})
    budget = max(measure(other), measure(created)) + headroom
    monkeypatch.setattr(context_store, limit, budget)
    assert measure({"id": _STUDY, **growth(large)}) < budget

    grown = context_store.upsert_context({"id": _STUDY, **growth(small)})
    with pytest.raises(context_store.StudyContextError) as caught:
        context_store.upsert_context({"id": _STUDY, **growth(large)})

    assert caught.value.detail["error"] == code
    assert caught.value.detail["study_context_id"] == _STUDY
    # The store reads as it did: the written study at its last revision, the
    # other untouched.
    listed = {row["id"]: row for row in context_store.list_contexts()["contexts"]}
    assert listed[_STUDY] == grown
    assert listed[_OTHER] == other
    assert context_store.get_context(_STUDY) == grown


def test_a_patch_past_the_budget_alone_keeps_its_own_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = context_store.upsert_context({"id": _STUDY, "question": "q"})
    budget = context_store._metadata_node_count(created) + 4
    monkeypatch.setattr(context_store, "_MAX_CONTEXT_NODES", budget)
    patch = {"id": _STUDY, **_modules(budget)}
    assert context_store._metadata_node_count(patch) > budget

    with pytest.raises(context_store.StudyContextError) as caught:
        context_store.upsert_context(patch)

    assert caught.value.detail["error"] == "study_context_too_complex"
    assert context_store.get_context(_STUDY) == created
