from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.figure2_canonical9.evaluator import input_binding_v2
from benchmarks.figure2_canonical9.evaluator.rubric_v1 import FIGURE2_TASK_IDS


def _tracked_payload() -> dict[str, object]:
    path = (
        Path(input_binding_v2.__file__).resolve().parents[1]
        / "canonical_run_input_bindings_v2.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def test_tracked_selector_is_exact_canonical9_and_blocked_until_owner_freeze() -> None:
    manifest, digest = input_binding_v2.load_canonical_run_input_bindings()

    assert tuple(item.task_id for item in manifest.tasks) == FIGURE2_TASK_IDS
    assert all(item.state == "blocked" for item in manifest.tasks)
    assert len(digest) == 64
    with pytest.raises(PermissionError, match="not input-frozen"):
        input_binding_v2.require_ready_task_binding(FIGURE2_TASK_IDS[0])


def test_selector_requires_canonical_json_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = tmp_path / "selector.json"
    selector.write_text(json.dumps(_tracked_payload(), indent=2), encoding="utf-8")
    monkeypatch.setattr(
        input_binding_v2,
        "_canonical_run_input_binding_path",
        lambda: selector,
    )

    with pytest.raises(
        input_binding_v2.CanonicalRunInputBindingError,
        match="canonical JSON",
    ):
        input_binding_v2.load_canonical_run_input_bindings()


def test_selector_rejects_duplicate_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = tmp_path / "selector.json"
    raw = json.dumps(_tracked_payload(), separators=(",", ":"))
    selector.write_text(
        raw.replace(
            '"schema_version":"easyicu.figure2_canonical_run_input_bindings/2"',
            '"schema_version":"easyicu.figure2_canonical_run_input_bindings/2",'
            '"schema_version":"easyicu.figure2_canonical_run_input_bindings/2"',
            1,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        input_binding_v2,
        "_canonical_run_input_binding_path",
        lambda: selector,
    )

    with pytest.raises(
        input_binding_v2.CanonicalRunInputBindingError,
        match="duplicate JSON key",
    ):
        input_binding_v2.load_canonical_run_input_bindings()


def test_selector_rejects_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(
        input_binding_v2._canonical_json_bytes(_tracked_payload()) + b"\n"
    )
    selector = tmp_path / "selector.json"
    selector.symlink_to(target)
    monkeypatch.setattr(
        input_binding_v2,
        "_canonical_run_input_binding_path",
        lambda: selector,
    )

    with pytest.raises(OSError):
        input_binding_v2.load_canonical_run_input_bindings()


def test_manifest_rejects_reordered_or_incomplete_suite() -> None:
    payload = _tracked_payload()
    tasks = list(payload["tasks"])
    payload["tasks"] = tuple(reversed(tasks))

    with pytest.raises(ValueError, match="exact Canonical9 order"):
        input_binding_v2.CanonicalRunInputBindingManifest.model_validate_json(
            json.dumps(payload),
            strict=True,
        )
