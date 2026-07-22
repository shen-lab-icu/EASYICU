from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.figure2_canonical9 import typed_export_seal as seal_mod
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


# --------------------------------------------------------------------------- #
# Retrofit review attestation: the real freeze -> load re-verification path
# --------------------------------------------------------------------------- #
def _mint_paper_ready_attestation(export: Path) -> dict:
    """Seal a paper-ready synthetic export (subject_id present, human-signed) and
    mint a real attestation through the gate."""

    export.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "subject_id": [10, 20, 30],
            "age": [65.0, 70.0, 55.0],
            "sex": ["Male", "Female", "Male"],
        }
    ).to_parquet(export / "demographics.parquet", index=False)
    pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "subject_id": [10, 20, 30],
            "charttime": [0.5, 2.0, 1.0],
            "lact": [1.2, 2.5, 3.1],
        }
    ).to_parquet(export / "blood_gas.parquet", index=False)
    (export / "easyicu_export_manifest.json").write_text(
        json.dumps({"database": "miiv"}), encoding="utf-8"
    )
    seal_mod.seal_export_structural_typed(export, value_vintage="20260717")
    # The real write-once Framework v2 HITL sign-off: driven through a genuine
    # LangGraph interrupt + checkpoint resume (not a hand-edited manifest flag).
    seal_mod.review_retrofit_export(
        export, reviewer="dr. reviewer", decided_at="2026-07-22"
    )
    return seal_mod.build_retrofit_review_attestation(export)


def _ready_selector_payload(*, attestation: dict | None) -> tuple[dict, str]:
    """A full 9-task selector with the first task frozen ready from a retrofit
    source; the remaining eight stay blocked."""

    payload = _tracked_payload()
    target = FIGURE2_TASK_IDS[0]
    ready = {
        "task_id": target,
        "state": "ready",
        "research_question_sha256": "a" * 64,
        "database": "miiv",
        "operational_exposure": None,
        "target_outcome": "in_hospital_mortality",
        "expected_run_input_capsule_schema_version": "easyicu.run_input_capsule/2",
        "scientific_identity_sha256": "b" * 64,
        "source_materialized_cohort_authority_ref": {
            "schema_version": "easyicu.materialized_cohort_authority/1",
            "file": "cohort.parquet",
            "sha256": "c" * 64,
            "size": 128,
        },
        "source_materialized_trajectory_authority_ref": None,
        "source_kind": "retrofit_sealed",
        "source_retrofit_review_attestation": attestation,
        "required_cohort_identity_policy": "unique_stay_per_patient",
    }
    payload["tasks"] = tuple(
        (
            ready
            if task["task_id"] == target
            else {
                "task_id": task["task_id"],
                "state": "blocked",
                "blockers": ["CANONICAL_INPUT_NOT_FROZEN"],
            }
        )
        for task in payload["tasks"]
    )
    return payload, target


def test_ready_retrofit_binding_reverifies_attestation_through_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "export_20260717"
    attestation = _mint_paper_ready_attestation(export)
    payload, target = _ready_selector_payload(attestation=attestation)
    selector = tmp_path / "selector.json"
    manifest = input_binding_v2.CanonicalRunInputBindingManifest.model_validate_json(
        json.dumps(payload), strict=True
    )
    selector.write_bytes(
        input_binding_v2._canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    )
    monkeypatch.setattr(
        input_binding_v2, "_canonical_run_input_binding_path", lambda: selector
    )
    _, binding, _, _ = input_binding_v2.require_ready_task_binding(
        target, source_export_dir=export
    )
    assert binding.source_kind == "retrofit_sealed"
    assert binding.source_retrofit_review_attestation.paper_ready is True


def test_retrofit_binding_without_source_dir_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "export_20260717"
    attestation = _mint_paper_ready_attestation(export)
    payload, target = _ready_selector_payload(attestation=attestation)
    selector = tmp_path / "selector.json"
    manifest = input_binding_v2.CanonicalRunInputBindingManifest.model_validate_json(
        json.dumps(payload), strict=True
    )
    selector.write_bytes(
        input_binding_v2._canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    )
    monkeypatch.setattr(
        input_binding_v2, "_canonical_run_input_binding_path", lambda: selector
    )
    # Neither a live export dir nor staged authority -> offline-only -> refuse.
    with pytest.raises(
        input_binding_v2.CanonicalRunInputBindingError, match="offline-only"
    ):
        input_binding_v2.require_ready_task_binding(target)


def test_retrofit_binding_missing_attestation_rejected_on_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, target = _ready_selector_payload(attestation=None)
    selector = tmp_path / "selector.json"
    # Hand-write canonical bytes for the invalid binding (it cannot be model-
    # serialized); the load path must reject it at validation time.
    selector.write_bytes(input_binding_v2._canonical_json_bytes(payload) + b"\n")
    monkeypatch.setattr(
        input_binding_v2, "_canonical_run_input_binding_path", lambda: selector
    )
    with pytest.raises(input_binding_v2.CanonicalRunInputBindingError):
        input_binding_v2.require_ready_task_binding(target)


def test_retrofit_binding_live_digest_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = tmp_path / "export_20260717"
    attestation = _mint_paper_ready_attestation(export)
    payload, target = _ready_selector_payload(attestation=attestation)
    selector = tmp_path / "selector.json"
    manifest = input_binding_v2.CanonicalRunInputBindingManifest.model_validate_json(
        json.dumps(payload), strict=True
    )
    selector.write_bytes(
        input_binding_v2._canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    )
    monkeypatch.setattr(
        input_binding_v2, "_canonical_run_input_binding_path", lambda: selector
    )
    # Tamper the source manifest after freezing the binding.
    manifest_path = export / "_manifest.json"
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["benign_marker"] = "tampered"
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        input_binding_v2.CanonicalRunInputBindingError, match="re-verification"
    ):
        input_binding_v2.require_ready_task_binding(target, source_export_dir=export)


# --------------------------------------------------------------------------- #
# Content-addressed staged acceptance: the production path (no live export dir)
# --------------------------------------------------------------------------- #
def _install_retrofit_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str]:
    """Seal+review an export, freeze a retrofit binding, and return (export, task)."""

    export = tmp_path / "export_20260717"
    attestation = _mint_paper_ready_attestation(export)
    payload, target = _ready_selector_payload(attestation=attestation)
    selector = tmp_path / "selector.json"
    manifest = input_binding_v2.CanonicalRunInputBindingManifest.model_validate_json(
        json.dumps(payload), strict=True
    )
    selector.write_bytes(
        input_binding_v2._canonical_json_bytes(manifest.model_dump(mode="json")) + b"\n"
    )
    monkeypatch.setattr(
        input_binding_v2, "_canonical_run_input_binding_path", lambda: selector
    )
    return export, target


def test_ready_retrofit_binding_reverifies_via_staged_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export, target = _install_retrofit_selector(tmp_path, monkeypatch)
    # Stage the content-addressed authority, then accept WITHOUT the export dir.
    staged = tmp_path / "run_dir" / "retrofit_source_authority"
    seal_mod.stage_retrofit_source_authority(export, staged)
    _, binding, _, _ = input_binding_v2.require_ready_task_binding(
        target, staged_authority_dir=staged
    )
    assert binding.source_kind == "retrofit_sealed"
    assert binding.required_cohort_identity_policy == "unique_stay_per_patient"


def test_retrofit_binding_staged_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export, target = _install_retrofit_selector(tmp_path, monkeypatch)
    staged = tmp_path / "run_dir" / "retrofit_source_authority"
    index = seal_mod.stage_retrofit_source_authority(export, staged)
    blob = staged / index["source_sidecar_blob"]
    blob.write_bytes(blob.read_bytes() + b" ")  # break content-address integrity
    with pytest.raises(
        input_binding_v2.CanonicalRunInputBindingError, match="re-verification"
    ):
        input_binding_v2.require_ready_task_binding(target, staged_authority_dir=staged)


def test_scoring_inputs_resolves_staged_authority_dir(tmp_path: Path) -> None:
    from benchmarks.figure2_canonical9.evaluator import scoring_inputs

    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    assert scoring_inputs._retrofit_staged_authority_dir(run_dir) is None
    (run_dir / scoring_inputs.RETROFIT_STAGED_AUTHORITY_DIR).mkdir()
    assert (
        scoring_inputs._retrofit_staged_authority_dir(run_dir)
        == run_dir / scoring_inputs.RETROFIT_STAGED_AUTHORITY_DIR
    )
