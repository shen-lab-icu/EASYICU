"""Crash/reopen contracts for the versioned EvidenceStore authority."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent import evidence as evidence_module
from easyicu.research_agent.evidence import DerivedFormulaError, EvidenceStore
from easyicu.research_agent.evidence_authority import (
    EVIDENCE_AUTHORITY_FILENAME,
    EVIDENCE_AUTHORITY_HEAD_FILENAME,
    EVIDENCE_AUTHORITY_MARKER_FILENAME,
    EVIDENCE_AUTHORITY_PREVIOUS_FILENAME,
    EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME,
    EVIDENCE_AUTHORITY_TRANSACTION_FILENAME,
    EvidenceAuthorityIntegrityError,
    load_current_evidence_snapshot,
)
from easyicu.research_agent.lock_authority import (
    LockAuthorityError,
    verified_unique_lock_anchor,
)


def _register_candidate(
    store: EvidenceStore,
    *,
    evidence_id: str = "candidate",
    publish_aliases: bool = False,
) -> object:
    return store.register_text(
        kind="statistic",
        description="Candidate result.",
        text='{"estimate": 1}',
        filename=f"{evidence_id}.json",
        produced_by_step="01_model",
        evidence_id=evidence_id,
        publish_aliases=publish_aliases,
    )


def _fail_named_write(
    monkeypatch: pytest.MonkeyPatch,
    *,
    filename: str,
) -> None:
    original = evidence_module._atomic_write_text

    def injected(path, text, **kwargs):
        if Path(path).name == filename:
            raise OSError(f"injected failure for {filename}")
        return original(path, text, **kwargs)

    monkeypatch.setattr(evidence_module, "_atomic_write_text", injected)


def test_corrupt_legacy_alias_ledger_fails_closed_across_reopen(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "evidence_index.json").write_text("[]", encoding="utf-8")
    (evidence_dir / "evidence_aliases.json").write_text("{broken", encoding="utf-8")

    for _ in range(2):
        with pytest.raises(EvidenceAuthorityIntegrityError, match="aliases"):
            EvidenceStore(tmp_path)


def test_verified_authority_repairs_corrupt_legacy_projection(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    record = _register_candidate(store, publish_aliases=True)
    aliases_path = tmp_path / "evidence" / "evidence_aliases.json"
    aliases_path.write_text("not-json", encoding="utf-8")

    reopened = EvidenceStore(tmp_path)

    assert reopened.get(record.evidence_id).evidence_id == record.evidence_id
    assert reopened.get("candidate").evidence_id == record.evidence_id
    assert json.loads(aliases_path.read_text(encoding="utf-8"))["candidate"] == (
        record.evidence_id
    )


def test_corrupt_current_authority_never_falls_back_to_flat_projection(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    _register_candidate(store, publish_aliases=True)
    authority_path = tmp_path / "evidence" / EVIDENCE_AUTHORITY_FILENAME
    authority_path.write_text("{}", encoding="utf-8")

    for _ in range(2):
        with pytest.raises(EvidenceAuthorityIntegrityError, match="authority"):
            EvidenceStore(tmp_path)


@pytest.mark.parametrize(
    "failed_filename",
    [
        EVIDENCE_AUTHORITY_TRANSACTION_FILENAME,
        EVIDENCE_AUTHORITY_PREVIOUS_FILENAME,
        "evidence_index.json",
        "evidence_aliases.json",
        "numeric_claims.json",
        EVIDENCE_AUTHORITY_FILENAME,
        EVIDENCE_AUTHORITY_HEAD_FILENAME,
        EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME,
    ],
)
def test_failed_promotion_reopens_previous_generation_without_alias_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_filename: str,
) -> None:
    store = EvidenceStore(tmp_path)
    record = _register_candidate(store)
    before = load_current_evidence_snapshot(tmp_path)
    original = evidence_module._atomic_write_text
    _fail_named_write(monkeypatch, filename=failed_filename)

    with pytest.raises(OSError, match="injected failure"):
        store.publish_step_success_aliases(
            {record.evidence_id: ["primary_association"]},
            step_id="01_model",
        )

    monkeypatch.setattr(evidence_module, "_atomic_write_text", original)
    reopened = EvidenceStore(tmp_path)
    after_failure = load_current_evidence_snapshot(tmp_path)
    assert after_failure.generation == before.generation
    assert after_failure.payload_sha256 == before.payload_sha256
    assert reopened.get("primary_association") is None
    assert reopened.get(record.evidence_id).metadata["aliases_published"] is False

    reopened.publish_step_success_aliases(
        {record.evidence_id: ["primary_association"]},
        step_id="01_model",
    )
    final = EvidenceStore(tmp_path)
    assert final.get("primary_association").evidence_id == record.evidence_id


@pytest.mark.parametrize(
    "failed_filename",
    [
        EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME,
        EVIDENCE_AUTHORITY_MARKER_FILENAME,
        EVIDENCE_AUTHORITY_TRANSACTION_FILENAME,
        EVIDENCE_AUTHORITY_FILENAME,
        EVIDENCE_AUTHORITY_HEAD_FILENAME,
    ],
)
def test_interrupted_initial_migration_reopens_baseline_and_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_filename: str,
) -> None:
    store = EvidenceStore(tmp_path)
    original = evidence_module._atomic_write_text
    _fail_named_write(monkeypatch, filename=failed_filename)

    with pytest.raises(OSError, match="injected failure"):
        _register_candidate(store, evidence_id="first")

    monkeypatch.setattr(evidence_module, "_atomic_write_text", original)
    reopened = EvidenceStore(tmp_path)
    assert reopened.records() == []
    _register_candidate(reopened, evidence_id="first")
    assert EvidenceStore(tmp_path).get("first") is not None


def test_stale_store_handle_cannot_overwrite_newer_generation(tmp_path: Path) -> None:
    first = EvidenceStore(tmp_path)
    stale = EvidenceStore(tmp_path)
    _register_candidate(first, evidence_id="first")

    with pytest.raises(EvidenceAuthorityIntegrityError, match="stale"):
        _register_candidate(stale, evidence_id="stale")

    reopened = EvidenceStore(tmp_path)
    assert reopened.get("first") is not None
    assert reopened.get("stale") is None


def test_exact_retry_does_not_advance_generation_or_stale_peer(
    tmp_path: Path,
) -> None:
    seeded = EvidenceStore(tmp_path)
    _register_candidate(seeded, evidence_id="seed", publish_aliases=True)
    first = EvidenceStore(tmp_path)
    peer = EvidenceStore(tmp_path)
    before = load_current_evidence_snapshot(tmp_path)

    _register_candidate(first, evidence_id="seed", publish_aliases=True)

    after_retry = load_current_evidence_snapshot(tmp_path)
    assert after_retry.generation == before.generation
    assert after_retry.payload_sha256 == before.payload_sha256
    _register_candidate(peer, evidence_id="peer")
    assert EvidenceStore(tmp_path).get("peer") is not None


def test_stale_handle_cannot_hide_drift_behind_exact_retry(tmp_path: Path) -> None:
    seeded = EvidenceStore(tmp_path)
    _register_candidate(seeded, evidence_id="seed", publish_aliases=True)
    stale = EvidenceStore(tmp_path)
    writer = EvidenceStore(tmp_path)
    _register_candidate(writer, evidence_id="newer")

    with pytest.raises(EvidenceAuthorityIntegrityError, match="stale"):
        _register_candidate(stale, evidence_id="seed", publish_aliases=True)


def test_stale_handle_cannot_overwrite_newer_selected_blob(tmp_path: Path) -> None:
    first = EvidenceStore(tmp_path)
    stale = EvidenceStore(tmp_path)
    selected = first.register_text(
        kind="log",
        description="Selected bytes.",
        text="selected",
        filename="same.txt",
        evidence_id="same",
    )

    with pytest.raises(EvidenceAuthorityIntegrityError, match="immutable payload"):
        stale.register_text(
            kind="log",
            description="Stale replacement.",
            text="stale replacement",
            filename="same.txt",
            evidence_id="same",
        )

    reopened = EvidenceStore(tmp_path)
    selected_path = tmp_path / selected.relative_path
    assert selected_path.read_text(encoding="utf-8") == "selected"
    assert reopened.get("same").sha256 == selected.sha256


@pytest.mark.parametrize("mode", ["raise", "keep_existing"])
def test_collision_policy_never_overwrites_selected_blob(
    tmp_path: Path,
    mode: str,
) -> None:
    store = EvidenceStore(tmp_path)
    original = store.register_text(
        kind="log",
        description="Original.",
        text="old bytes",
        filename="same.txt",
        evidence_id="same",
    )
    selected = tmp_path / original.relative_path

    if mode == "raise":
        with pytest.raises(ValueError, match="collision"):
            store.register_text(
                kind="log",
                description="Replacement.",
                text="new bytes",
                filename="same.txt",
                evidence_id="same",
                on_sha_change=mode,
            )
    else:
        returned = store.register_text(
            kind="log",
            description="Replacement.",
            text="new bytes",
            filename="same.txt",
            evidence_id="same",
            on_sha_change=mode,
        )
        assert returned.sha256 == original.sha256

    assert selected.read_text(encoding="utf-8") == "old bytes"
    assert EvidenceStore(tmp_path).get("same").sha256 == original.sha256


@pytest.mark.parametrize("include_numeric_claims", [False, True])
def test_valid_flat_legacy_store_migrates_only_on_first_mutation(
    tmp_path: Path,
    include_numeric_claims: bool,
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    legacy_record = {
        "evidence_id": "legacy",
        "kind": "log",
        "description": "Minimal pre-generation record.",
        "relative_path": "evidence/legacy__legacy.txt",
        "sha256": "a" * 64,
    }
    (evidence_dir / "evidence_index.json").write_text(
        json.dumps([legacy_record]), encoding="utf-8"
    )
    (evidence_dir / "evidence_aliases.json").write_text(
        json.dumps({"legacy": "legacy"}), encoding="utf-8"
    )
    if include_numeric_claims:
        (evidence_dir / "numeric_claims.json").write_text("[]", encoding="utf-8")

    legacy = EvidenceStore(tmp_path)
    assert legacy.get("legacy").inputs == []
    assert not (evidence_dir / EVIDENCE_AUTHORITY_FILENAME).exists()

    _register_candidate(legacy, evidence_id="modern")
    assert (evidence_dir / EVIDENCE_AUTHORITY_FILENAME).is_file()
    reopened = EvidenceStore(tmp_path)
    assert {record.evidence_id for record in reopened.records()} == {
        "legacy",
        "modern",
    }


@pytest.mark.parametrize(
    "present_files",
    [
        ("evidence_index.json",),
        ("evidence_aliases.json",),
        ("numeric_claims.json",),
        ("evidence_index.json", "numeric_claims.json"),
    ],
)
def test_incomplete_flat_legacy_layout_fails_closed(
    tmp_path: Path,
    present_files: tuple[str, ...],
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    payloads = {
        "evidence_index.json": "[]",
        "evidence_aliases.json": "{}",
        "numeric_claims.json": "[]",
    }
    for filename in present_files:
        (evidence_dir / filename).write_text(payloads[filename], encoding="utf-8")

    with pytest.raises(EvidenceAuthorityIntegrityError, match="incomplete"):
        EvidenceStore(tmp_path)


def test_missing_modern_authority_never_falls_back_to_changed_projection(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    _register_candidate(store, evidence_id="selected", publish_aliases=True)
    evidence_dir = tmp_path / "evidence"
    (evidence_dir / EVIDENCE_AUTHORITY_FILENAME).unlink()

    with pytest.raises(EvidenceAuthorityIntegrityError, match="missing"):
        EvidenceStore(tmp_path)


def test_modern_authority_requires_permanent_format_marker(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    _register_candidate(store)
    (tmp_path / "evidence" / EVIDENCE_AUTHORITY_MARKER_FILENAME).unlink()

    with pytest.raises(EvidenceAuthorityIntegrityError, match="files are missing"):
        EvidenceStore(tmp_path)


def test_deleting_inner_authority_cannot_downgrade_modern_store(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    _register_candidate(store, evidence_id="selected", publish_aliases=True)
    evidence_dir = tmp_path / "evidence"
    for filename in (
        EVIDENCE_AUTHORITY_FILENAME,
        EVIDENCE_AUTHORITY_PREVIOUS_FILENAME,
        EVIDENCE_AUTHORITY_MARKER_FILENAME,
        "evidence_index.json",
        "evidence_aliases.json",
        "numeric_claims.json",
    ):
        path = evidence_dir / filename
        if path.exists():
            path.unlink()
    (tmp_path / EVIDENCE_AUTHORITY_HEAD_FILENAME).unlink()

    with pytest.raises(EvidenceAuthorityIntegrityError, match="head is missing"):
        EvidenceStore(tmp_path)


def test_old_head_and_authority_cannot_rollback_newer_root_high_water(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    _register_candidate(store, evidence_id="first")
    authority_path = tmp_path / "evidence" / EVIDENCE_AUTHORITY_FILENAME
    head_path = tmp_path / EVIDENCE_AUTHORITY_HEAD_FILENAME
    old_authority = authority_path.read_bytes()
    old_head = head_path.read_bytes()
    _register_candidate(store, evidence_id="second")

    authority_path.write_bytes(old_authority)
    head_path.write_bytes(old_head)

    with pytest.raises(
        EvidenceAuthorityIntegrityError, match="disagree or were rolled back"
    ):
        EvidenceStore(tmp_path)


def test_old_root_marker_alone_cannot_impersonate_interrupted_commit(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    _register_candidate(store, evidence_id="first")
    root_marker_path = tmp_path / EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME
    old_root_marker = root_marker_path.read_bytes()
    _register_candidate(store, evidence_id="second")
    receipt = json.loads(
        (tmp_path / EVIDENCE_AUTHORITY_TRANSACTION_FILENAME).read_text(encoding="utf-8")
    )
    assert receipt["state"] == "committed"

    root_marker_path.write_bytes(old_root_marker)

    with pytest.raises(EvidenceAuthorityIntegrityError, match="committed.*rolled back"):
        EvidenceStore(tmp_path)


def test_partial_bootstrap_repairs_corrupt_inner_marker_before_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvidenceStore(tmp_path)
    original = evidence_module._atomic_write_text
    _fail_named_write(monkeypatch, filename=EVIDENCE_AUTHORITY_HEAD_FILENAME)
    with pytest.raises(OSError, match="injected failure"):
        _register_candidate(store, evidence_id="first")
    monkeypatch.setattr(evidence_module, "_atomic_write_text", original)

    marker_path = tmp_path / "evidence" / EVIDENCE_AUTHORITY_MARKER_FILENAME
    marker_path.write_text("{}", encoding="utf-8")
    reopened = EvidenceStore(tmp_path)
    _register_candidate(reopened, evidence_id="first")

    assert EvidenceStore(tmp_path).get("first") is not None


def test_partial_bootstrap_rejects_symlinked_inner_marker_before_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not hasattr(Path, "symlink_to"):
        pytest.skip("symlinks unavailable")
    store = EvidenceStore(tmp_path)
    original = evidence_module._atomic_write_text
    _fail_named_write(monkeypatch, filename=EVIDENCE_AUTHORITY_HEAD_FILENAME)
    with pytest.raises(OSError, match="injected failure"):
        _register_candidate(store, evidence_id="first")
    monkeypatch.setattr(evidence_module, "_atomic_write_text", original)

    marker_path = tmp_path / "evidence" / EVIDENCE_AUTHORITY_MARKER_FILENAME
    marker_path.unlink()
    marker_path.symlink_to(tmp_path / "outside-marker.json")
    reopened = EvidenceStore(tmp_path)
    with pytest.raises(EvidenceAuthorityIntegrityError, match="symbolic link"):
        _register_candidate(reopened, evidence_id="first")
    assert not (tmp_path / EVIDENCE_AUTHORITY_HEAD_FILENAME).exists()


def test_post_replace_bootstrap_head_failure_survives_repeated_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvidenceStore(tmp_path)
    original = evidence_module._atomic_write_text
    injected = False

    def write_head_then_fail(path, text, **kwargs):
        nonlocal injected
        result = original(path, text, **kwargs)
        if not injected and Path(path).name == EVIDENCE_AUTHORITY_HEAD_FILENAME:
            injected = True
            raise OSError("bootstrap head acknowledgement failure")
        return result

    monkeypatch.setattr(evidence_module, "_atomic_write_text", write_head_then_fail)
    with pytest.raises(OSError, match="bootstrap head acknowledgement failure"):
        _register_candidate(store, evidence_id="first")
    monkeypatch.setattr(evidence_module, "_atomic_write_text", original)

    assert EvidenceStore(tmp_path).records() == []
    assert EvidenceStore(tmp_path).records() == []
    _register_candidate(EvidenceStore(tmp_path), evidence_id="first")
    assert EvidenceStore(tmp_path).get("first") is not None


def test_post_replace_bootstrap_root_failure_survives_repeated_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvidenceStore(tmp_path)
    original = evidence_module._atomic_write_text
    injected = False

    def write_selected_root_then_fail(path, text, **kwargs):
        nonlocal injected
        result = original(path, text, **kwargs)
        payload = (
            json.loads(text)
            if Path(path).name == EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME
            else {}
        )
        if not injected and payload.get("selected_generation") == 0:
            injected = True
            raise OSError("bootstrap root acknowledgement failure")
        return result

    monkeypatch.setattr(
        evidence_module,
        "_atomic_write_text",
        write_selected_root_then_fail,
    )
    with pytest.raises(OSError, match="bootstrap root acknowledgement failure"):
        _register_candidate(store, evidence_id="first")
    monkeypatch.setattr(evidence_module, "_atomic_write_text", original)

    assert EvidenceStore(tmp_path).records() == []
    assert EvidenceStore(tmp_path).records() == []
    _register_candidate(EvidenceStore(tmp_path), evidence_id="first")
    assert EvidenceStore(tmp_path).get("first") is not None


def test_bootstrap_committed_receipt_prewrite_failure_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvidenceStore(tmp_path)
    original = evidence_module._atomic_write_text
    injected = False

    def fail_bootstrap_commit(path, text, **kwargs):
        nonlocal injected
        payload = (
            json.loads(text)
            if Path(path).name == EVIDENCE_AUTHORITY_TRANSACTION_FILENAME
            else {}
        )
        if (
            not injected
            and payload.get("state") == "committed"
            and payload.get("candidate_generation") == 0
        ):
            injected = True
            raise OSError("bootstrap commit receipt failure")
        return original(path, text, **kwargs)

    monkeypatch.setattr(evidence_module, "_atomic_write_text", fail_bootstrap_commit)
    with pytest.raises(OSError, match="bootstrap commit receipt failure"):
        _register_candidate(store, evidence_id="first")
    monkeypatch.setattr(evidence_module, "_atomic_write_text", original)

    assert EvidenceStore(tmp_path).records() == []
    assert EvidenceStore(tmp_path).records() == []
    _register_candidate(EvidenceStore(tmp_path), evidence_id="first")
    assert EvidenceStore(tmp_path).get("first") is not None


def test_copy_rejects_source_bytes_that_do_not_match_precomputed_digest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.csv"
    source.write_text("changed\n", encoding="utf-8")
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    target = evidence_dir / "copy.csv"

    with pytest.raises(EvidenceAuthorityIntegrityError, match="source changed"):
        evidence_module._atomic_copy_file(
            source,
            target,
            expected_root=tmp_path,
            expected_sha256=evidence_module.sha256_of_bytes(b"original\n"),
        )

    assert not target.exists()


def test_existing_empty_evidence_ledger_is_not_treated_as_legacy_without_anchor(
    tmp_path: Path,
) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    (evidence_dir / "evidence_index.json").write_text("[]", encoding="utf-8")
    (evidence_dir / "evidence_aliases.json").write_text("{}", encoding="utf-8")

    with pytest.raises(LockAuthorityError, match="no unique"):
        verified_unique_lock_anchor(
            run_dir=tmp_path,
            evidence_id="analysis_lock",
            label="analysis lock",
        )


def test_step_summary_numeric_batch_commits_once(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    record = _register_candidate(store, evidence_id="summary")
    before = load_current_evidence_snapshot(tmp_path)

    claims = store.register_step_summary_numerics(
        step_id="01_model",
        evidence_id=record.evidence_id,
        summary={"estimate": 1.25, "ci": {"low": 1.0, "high": 1.5}},
    )

    after = load_current_evidence_snapshot(tmp_path)
    assert len(claims) == 3
    assert len(after.numeric_claims) == 3
    assert after.generation == int(before.generation or 0) + 1


def test_step_summary_numeric_batch_failure_restores_whole_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvidenceStore(tmp_path)
    record = _register_candidate(store, evidence_id="summary")
    before = load_current_evidence_snapshot(tmp_path)
    original = evidence_module._atomic_write_text
    _fail_named_write(monkeypatch, filename=EVIDENCE_AUTHORITY_FILENAME)

    with pytest.raises(OSError, match="injected failure"):
        store.register_step_summary_numerics(
            step_id="01_model",
            evidence_id=record.evidence_id,
            summary={"estimate": 1.25, "ci": {"low": 1.0, "high": 1.5}},
        )

    assert store.numeric_claims() == []
    monkeypatch.setattr(evidence_module, "_atomic_write_text", original)
    reopened = EvidenceStore(tmp_path)
    after = load_current_evidence_snapshot(tmp_path)
    assert after.generation == before.generation
    assert after.payload_sha256 == before.payload_sha256
    assert reopened.numeric_claims() == []


def test_failed_success_promotion_rolls_back_numeric_claims_and_blocks_laundering(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    owner = store.register_text(
        kind="statistic",
        description="Selected owner.",
        text='{"estimate": 1}',
        filename="owner.json",
        produced_by_step="01_owner",
        evidence_id="owner",
        publish_aliases=False,
    )
    store.publish_step_success_aliases(
        {owner.evidence_id: ["primary_association"]},
        step_id="01_owner",
    )
    failed = store.register_text(
        kind="statistic",
        description="Failed challenger.",
        text='{"estimate": 9}',
        filename="failed.json",
        produced_by_step="02_failed",
        evidence_id="failed",
        publish_aliases=False,
    )

    with pytest.raises(ValueError, match="already owned"):
        with store.success_publication_transaction():
            store.register_step_summary_numerics(
                step_id="02_failed",
                evidence_id=failed.evidence_id,
                summary={"estimate": 9},
            )
            store.publish_step_success_aliases(
                {failed.evidence_id: ["primary_association"]},
                step_id="02_failed",
            )

    assert not [
        claim for claim in store.numeric_claims() if claim.step_id == "02_failed"
    ]
    with pytest.raises(DerivedFormulaError, match="not found in registry"):
        store.register_derived_claim(
            name="laundered_estimate",
            formula="failed_estimate * 2",
            explanation="Must not consume a failed attempt.",
            sources={"failed_estimate": ("02_failed", "estimate")},
            evidence_id="03_downstream",
            step_id="03_downstream",
        )


def test_publication_transaction_rolls_back_on_stale_generation(tmp_path: Path) -> None:
    seeded = EvidenceStore(tmp_path)
    seed = _register_candidate(seeded, evidence_id="seed")
    stale = EvidenceStore(tmp_path)
    writer = EvidenceStore(tmp_path)
    _register_candidate(writer, evidence_id="newer")

    with pytest.raises(EvidenceAuthorityIntegrityError, match="stale"):
        with stale.success_publication_transaction():
            stale.register_step_summary_numerics(
                step_id="01_model",
                evidence_id=seed.evidence_id,
                summary={"estimate": 4.2},
            )

    assert stale.numeric_claims() == []
    assert EvidenceStore(tmp_path).numeric_claims() == []


def test_nested_publication_inner_failure_can_be_caught_without_partial_state(
    tmp_path: Path,
) -> None:
    store = EvidenceStore(tmp_path)
    record = _register_candidate(store, evidence_id="summary")
    before = load_current_evidence_snapshot(tmp_path)

    with store.success_publication_transaction():
        store.register_step_summary_numerics(
            step_id="01_model",
            evidence_id=record.evidence_id,
            summary={"kept_before": 1},
        )
        try:
            with store.success_publication_transaction():
                store.register_step_summary_numerics(
                    step_id="01_model",
                    evidence_id=record.evidence_id,
                    summary={"rolled_back": 2},
                )
                raise RuntimeError("abort inner scope")
        except RuntimeError:
            pass
        store.register_step_summary_numerics(
            step_id="01_model",
            evidence_id=record.evidence_id,
            summary={"kept_after": 3},
        )

    after = load_current_evidence_snapshot(tmp_path)
    fields = {claim.source_field for claim in store.numeric_claims()}
    assert fields == {"kept_before", "kept_after"}
    assert after.generation == int(before.generation or 0) + 1


def test_numeric_and_alias_publication_share_root_commit_failure_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvidenceStore(tmp_path)
    record = _register_candidate(store, evidence_id="summary")
    original = evidence_module._atomic_write_text
    _fail_named_write(
        monkeypatch,
        filename=EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME,
    )

    with pytest.raises(OSError, match="injected failure"):
        with store.success_publication_transaction():
            store.register_step_summary_numerics(
                step_id="01_model",
                evidence_id=record.evidence_id,
                summary={"estimate": 2.5},
            )
            store.publish_step_success_aliases(
                {record.evidence_id: ["primary_association"]},
                step_id="01_model",
            )

    monkeypatch.setattr(evidence_module, "_atomic_write_text", original)
    reopened = EvidenceStore(tmp_path)
    assert reopened.numeric_claims() == []
    assert reopened.get("primary_association") is None


def test_post_replace_staging_root_error_does_not_publish_failed_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvidenceStore(tmp_path)
    record = _register_candidate(store, evidence_id="summary")
    original = evidence_module._atomic_write_text
    injected = False

    def write_then_fail(path, text, **kwargs):
        nonlocal injected
        result = original(path, text, **kwargs)
        if not injected and Path(path).name == EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME:
            injected = True
            raise OSError("post-replace acknowledgement failure")
        return result

    monkeypatch.setattr(evidence_module, "_atomic_write_text", write_then_fail)
    with pytest.raises(OSError, match="post-replace acknowledgement failure"):
        with store.success_publication_transaction():
            store.register_step_summary_numerics(
                step_id="01_model",
                evidence_id=record.evidence_id,
                summary={"estimate": 2.5},
            )
            store.publish_step_success_aliases(
                {record.evidence_id: ["primary_association"]},
                step_id="01_model",
            )

    assert injected is True
    reopened = EvidenceStore(tmp_path)
    assert reopened.get("primary_association") is None
    assert reopened.numeric_claims() == []


def test_post_replace_commit_receipt_error_reconciles_as_committed_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvidenceStore(tmp_path)
    record = _register_candidate(store, evidence_id="summary")
    original = evidence_module._atomic_write_text
    injected = False

    def commit_then_fail(path, text, **kwargs):
        nonlocal injected
        result = original(path, text, **kwargs)
        payload = (
            json.loads(text)
            if Path(path).name == EVIDENCE_AUTHORITY_TRANSACTION_FILENAME
            else {}
        )
        if not injected and payload.get("state") == "committed":
            injected = True
            raise OSError("post-commit acknowledgement failure")
        return result

    monkeypatch.setattr(evidence_module, "_atomic_write_text", commit_then_fail)
    with store.success_publication_transaction():
        store.register_step_summary_numerics(
            step_id="01_model",
            evidence_id=record.evidence_id,
            summary={"estimate": 2.5},
        )
        store.publish_step_success_aliases(
            {record.evidence_id: ["primary_association"]},
            step_id="01_model",
        )

    assert injected is True
    reopened = EvidenceStore(tmp_path)
    assert reopened.get("primary_association").evidence_id == record.evidence_id
    assert [claim.source_field for claim in reopened.numeric_claims()] == ["estimate"]


def test_commit_receipt_prewrite_failure_keeps_candidate_unpublished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = EvidenceStore(tmp_path)
    record = _register_candidate(store, evidence_id="summary")
    original = evidence_module._atomic_write_text
    injected = False

    def fail_commit(path, text, **kwargs):
        nonlocal injected
        payload = (
            json.loads(text)
            if Path(path).name == EVIDENCE_AUTHORITY_TRANSACTION_FILENAME
            else {}
        )
        if not injected and payload.get("state") == "committed":
            injected = True
            raise OSError("commit receipt prewrite failure")
        return original(path, text, **kwargs)

    monkeypatch.setattr(evidence_module, "_atomic_write_text", fail_commit)
    with pytest.raises(OSError, match="commit receipt prewrite failure"):
        with store.success_publication_transaction():
            store.register_step_summary_numerics(
                step_id="01_model",
                evidence_id=record.evidence_id,
                summary={"estimate": 2.5},
            )
            store.publish_step_success_aliases(
                {record.evidence_id: ["primary_association"]},
                step_id="01_model",
            )

    monkeypatch.setattr(evidence_module, "_atomic_write_text", original)
    reopened = EvidenceStore(tmp_path)
    assert reopened.get("primary_association") is None
    assert reopened.numeric_claims() == []


@pytest.mark.parametrize(
    "recovery_failure",
    [EVIDENCE_AUTHORITY_HEAD_FILENAME, EVIDENCE_AUTHORITY_FILENAME],
)
def test_interrupted_recovery_remains_retryable_at_each_repair_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_failure: str,
) -> None:
    store = EvidenceStore(tmp_path)
    record = _register_candidate(store, evidence_id="summary")
    original = evidence_module._atomic_write_text
    _fail_named_write(
        monkeypatch,
        filename=EVIDENCE_AUTHORITY_ROOT_MARKER_FILENAME,
    )
    with pytest.raises(OSError, match="injected failure"):
        store.publish_step_success_aliases(
            {record.evidence_id: ["primary_association"]},
            step_id="01_model",
        )

    monkeypatch.setattr(evidence_module, "_atomic_write_text", original)
    _fail_named_write(monkeypatch, filename=recovery_failure)
    with pytest.raises(OSError, match="injected failure"):
        EvidenceStore(tmp_path)

    monkeypatch.setattr(evidence_module, "_atomic_write_text", original)
    reopened = EvidenceStore(tmp_path)
    assert reopened.get("primary_association") is None
    assert reopened.get(record.evidence_id) is not None


def test_file_registration_removes_publish_temporary_file(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n", encoding="utf-8")

    EvidenceStore(tmp_path).register_file(
        kind="table",
        description="Immutable source copy.",
        source_path=source,
        evidence_id="source_table",
    )

    assert not list((tmp_path / "evidence").glob(".*.tmp"))
