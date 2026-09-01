"""Fail-closed report-only resume contracts for Writer drafts."""

from __future__ import annotations

from pathlib import Path

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.reporting.write_phase import (
    _verified_resume_writer_scaffold,
    _verified_resume_writer_scaffold_for_quality_migration,
)
from easyicu.research_agent.reporting.manuscript_post import (
    _remove_unregistered_evidence_placeholders,
)
from easyicu.research_agent.reporting.manuscript_sections import (
    manuscript_writer_contract_sha256,
)


def _step_records(*, attempt_id: str = "run:model:1") -> list[dict[str, object]]:
    return [
        {
            "step_id": "model",
            "attempt_id": attempt_id,
            "status": "ok",
            "evidence_ids": ["model_result"],
        }
    ]


def _registered_scaffold(
    tmp_path: Path,
    *,
    writer_contract_sha256: str | None = None,
) -> tuple[EvidenceStore, object, str]:
    scaffold = "## Results\n\nAUROC was 0.76 {evidence:model_result}.\n"
    source = tmp_path / "prior_scaffold.md"
    source.write_text(scaffold, encoding="utf-8")
    evidence = EvidenceStore(tmp_path)
    record = evidence.register_file(
        kind="log",
        description="Prior raw Writer scaffold.",
        source_path=source,
        evidence_id="manuscript_scaffold_raw",
        producer="writer",
        generation_mode="llm",
        metadata={
            "writer_contract_sha256": (
                writer_contract_sha256 or manuscript_writer_contract_sha256()
            )
        },
    )
    return evidence, record, scaffold


def test_report_only_resume_reuses_digest_verified_prior_writer_scaffold(
    tmp_path: Path,
) -> None:
    evidence, record, scaffold = _registered_scaffold(tmp_path)
    records = _step_records()

    reused = _verified_resume_writer_scaffold(
        resume_state={"per_step_records": records},
        evidence=evidence,
        run_dir=tmp_path,
        per_step_records=records,
    )

    assert reused is not None
    text, detail = reused
    assert text == scaffold
    assert detail["reason_code"] == "verified_prior_writer_scaffold_reused"
    assert detail["source_sha256"] == record.sha256
    assert detail["writer_contract_sha256"] == manuscript_writer_contract_sha256()


def test_resume_does_not_reuse_writer_scaffold_from_an_older_contract(
    tmp_path: Path,
) -> None:
    evidence, _, _ = _registered_scaffold(
        tmp_path,
        writer_contract_sha256="0" * 64,
    )
    records = _step_records()

    reused = _verified_resume_writer_scaffold(
        resume_state={"per_step_records": records},
        evidence=evidence,
        run_dir=tmp_path,
        per_step_records=records,
    )

    assert reused is None

    migration = _verified_resume_writer_scaffold_for_quality_migration(
        resume_state={"per_step_records": records},
        evidence=evidence,
        run_dir=tmp_path,
        per_step_records=records,
    )
    assert migration is not None
    text, detail = migration
    assert text == "## Results\n\nAUROC was 0.76 {evidence:model_result}.\n"
    assert detail["reason_code"] == ("verified_prior_writer_scaffold_quality_migration")


def test_resume_reuses_latest_matching_versioned_writer_scaffold(
    tmp_path: Path,
) -> None:
    evidence, _, _ = _registered_scaffold(
        tmp_path,
        writer_contract_sha256="0" * 64,
    )
    latest_text = "## Results\n\nAUROC was 0.79 {evidence:model_result}.\n"
    latest_path = tmp_path / "latest_scaffold.md"
    latest_path.write_text(latest_text, encoding="utf-8")
    latest = evidence.register_file(
        kind="log",
        description="Regenerated raw Writer scaffold.",
        source_path=latest_path,
        evidence_id="manuscript_scaffold_raw",
        producer="writer",
        generation_mode="llm",
        metadata={
            "writer_contract_sha256": manuscript_writer_contract_sha256(),
        },
        on_sha_change="new_id",
    )
    records = _step_records()

    reused = _verified_resume_writer_scaffold(
        resume_state={"per_step_records": records},
        evidence=evidence,
        run_dir=tmp_path,
        per_step_records=records,
    )

    assert reused is not None
    text, detail = reused
    assert text == latest_text
    assert detail["source_evidence_id"] == latest.evidence_id


def test_resume_does_not_reuse_writer_scaffold_after_step_reexecution(
    tmp_path: Path,
) -> None:
    evidence, _, _ = _registered_scaffold(tmp_path)

    reused = _verified_resume_writer_scaffold(
        resume_state={"per_step_records": _step_records(attempt_id="run:model:1")},
        evidence=evidence,
        run_dir=tmp_path,
        per_step_records=_step_records(attempt_id="run:model:2"),
    )

    assert reused is None


def test_resume_does_not_reuse_tampered_writer_scaffold(tmp_path: Path) -> None:
    evidence, record, _ = _registered_scaffold(tmp_path)
    (tmp_path / record.relative_path).write_text("tampered", encoding="utf-8")
    records = _step_records()

    reused = _verified_resume_writer_scaffold(
        resume_state={"per_step_records": records},
        evidence=evidence,
        run_dir=tmp_path,
        per_step_records=records,
    )

    assert reused is None


def test_unregistered_placeholder_is_removed_but_registered_citation_remains() -> None:
    text = "Calibration slope was 0.98 {evidence:stale} {evidence:registered}."

    repaired, removed = _remove_unregistered_evidence_placeholders(
        text,
        allowed_evidence_ids=["registered"],
    )

    assert removed == ["stale"]
    assert "{evidence:stale}" not in repaired
    assert repaired.count("{evidence:registered}") == 1


def test_unregistered_only_placeholder_does_not_gain_replacement_authority() -> None:
    text = "Unsupported result was 1.23 {evidence:stale}."

    repaired, removed = _remove_unregistered_evidence_placeholders(
        text,
        allowed_evidence_ids=["registered"],
    )

    assert removed == ["stale"]
    assert repaired == "Unsupported result was 1.23."
    assert "{evidence:" not in repaired
