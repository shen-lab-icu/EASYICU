from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from easyicu.research_agent.know_how.registry import (
    reviewable_card_content_sha256,
)

from benchmarks.figure2_canonical9.scientific_protocol_authority import (
    REQUIRED_SCIENTIFIC_PROTOCOLS,
    ScientificProtocolAuthority,
    ScientificProtocolAuthorityError,
    ScientificProtocolTaskBinding,
    load_verified_scientific_protocol_authority,
)
from benchmarks.figure2_canonical9.case_scientific_protocol import (
    build_runtime_scientific_projection,
    case_protocol_content_sha256,
    default_case_protocol_path,
    load_case_scientific_protocol,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _reviewed_card(
    tmp_path: Path,
    *,
    task_id: str,
    card_id: str,
) -> ScientificProtocolTaskBinding:
    source = (
        _REPO_ROOT / "src/easyicu/data/research_know_how" / f"{card_id}.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["review_status"] = "clinical_reviewed"
    payload["review_attestation"] = None
    reviewed_content_sha256 = reviewable_card_content_sha256(payload)
    protocol_path = tmp_path / f"{task_id}.protocol.json"
    protocol_path.write_bytes(default_case_protocol_path(task_id).read_bytes())
    protocol = load_case_scientific_protocol(
        protocol_path,
        expected_task_id=task_id,
    )
    protocol_content_sha256 = case_protocol_content_sha256(protocol)
    runtime_projection_sha256 = build_runtime_scientific_projection(
        protocol
    ).runtime_projection_sha256
    payload["review_attestation"] = {
        "reviewer_owner": "Synthetic clinical-and-methods test board",
        "review_date": "2026-07-26",
        "card_version": payload["version"],
        "reviewed_content_sha256": reviewed_content_sha256,
        "protocol_content_sha256": protocol_content_sha256,
        "runtime_projection_sha256": runtime_projection_sha256,
        "review_scope": ["clinical protocol", "statistical methods"],
        "literature_search_cutoff": "2026-07-25",
        "clinical_reviewed": True,
        "methods_reviewed": True,
    }
    path = tmp_path / f"{task_id}.reviewed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ScientificProtocolTaskBinding(
        task_id=task_id,
        card_id=card_id,
        card_version=payload["version"],
        card_path=str(path),
        card_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reviewed_content_sha256=reviewed_content_sha256,
        protocol_path=str(protocol_path),
        protocol_file_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        protocol_content_sha256=protocol_content_sha256,
        runtime_projection_sha256=runtime_projection_sha256,
    )


def _authority(tmp_path: Path) -> tuple[Path, ScientificProtocolAuthority]:
    bindings = [
        _reviewed_card(tmp_path, task_id=task_id, card_id=card_id)
        for task_id, card_id in REQUIRED_SCIENTIFIC_PROTOCOLS
    ]
    authority = ScientificProtocolAuthority.build(tasks=bindings)
    path = tmp_path / "scientific_protocol_authority.json"
    path.write_text(authority.model_dump_json(), encoding="utf-8")
    return path, authority


def test_exact_dual_reviewed_protocol_authority_verifies(tmp_path: Path) -> None:
    path, expected = _authority(tmp_path)

    loaded, file_sha256 = load_verified_scientific_protocol_authority(path)

    assert loaded == expected
    assert file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert tuple((task.task_id, task.card_id) for task in loaded.tasks) == (
        REQUIRED_SCIENTIFIC_PROTOCOLS
    )


def test_reviewed_card_tamper_invalidates_authority(tmp_path: Path) -> None:
    path, authority = _authority(tmp_path)
    card_path = Path(authority.tasks[0].card_path)
    card_path.write_text(
        card_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ScientificProtocolAuthorityError, match="file digest"):
        load_verified_scientific_protocol_authority(path)


def test_reviewed_case_protocol_tamper_invalidates_authority(tmp_path: Path) -> None:
    path, authority = _authority(tmp_path)
    protocol_path = Path(authority.tasks[1].protocol_path)
    protocol_path.write_text(
        protocol_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ScientificProtocolAuthorityError, match="protocol file digest"):
        load_verified_scientific_protocol_authority(path)


def test_curated_mvp_card_cannot_masquerade_as_formal_review(
    tmp_path: Path,
) -> None:
    path, authority = _authority(tmp_path)
    source = (
        _REPO_ROOT
        / "src/easyicu/data/research_know_how"
        / "early_peak_lactate_association.json"
    )
    curated = tmp_path / "curated.json"
    curated.write_bytes(source.read_bytes())
    payload = json.loads(curated.read_text(encoding="utf-8"))
    replacement = authority.tasks[0].model_copy(
        update={
            "card_path": str(curated),
            "card_file_sha256": hashlib.sha256(curated.read_bytes()).hexdigest(),
            "reviewed_content_sha256": reviewable_card_content_sha256(payload),
        }
    )
    replaced = ScientificProtocolAuthority.build(
        tasks=(replacement, *authority.tasks[1:])
    )
    path.write_text(replaced.model_dump_json(), encoding="utf-8")

    with pytest.raises(
        ScientificProtocolAuthorityError,
        match="lacks formal clinical-and-methods attestation",
    ):
        load_verified_scientific_protocol_authority(path)


def test_authority_rejects_wrong_order_duplicate_json_and_symlink(
    tmp_path: Path,
) -> None:
    path, authority = _authority(tmp_path)
    with pytest.raises(ValueError, match="exact ordered"):
        ScientificProtocolAuthority.build(tasks=tuple(reversed(authority.tasks)))

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"easyicu.figure2_scientific_protocol_authority/3",'
        '"schema_version":"easyicu.figure2_scientific_protocol_authority/3",'
        '"tasks":[],"authority_digest":"' + "0" * 64 + '"}',
        encoding="utf-8",
    )
    with pytest.raises(ScientificProtocolAuthorityError, match="strict JSON"):
        load_verified_scientific_protocol_authority(duplicate)

    alias = tmp_path / "authority-link.json"
    alias.symlink_to(path)
    with pytest.raises(ScientificProtocolAuthorityError, match="non-symlink"):
        load_verified_scientific_protocol_authority(alias)
