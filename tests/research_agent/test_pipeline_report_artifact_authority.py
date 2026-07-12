"""Fail-closed provenance checks at publication-readiness boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.evidence import EvidenceStore
from easyicu.research_agent.pipeline_report import (
    _latest_publication_figure_audit_status,
    _publication_figure_bundle_ready,
)
from easyicu.research_agent.publication_figures import (
    PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
)


def _publication_bundle(tmp_path: Path) -> tuple[EvidenceStore, dict[str, Path]]:
    evidence = EvidenceStore(tmp_path)
    source = tmp_path / "source.csv"
    source.write_text("term,estimate\nexposure,1.2\n", encoding="utf-8")
    source_record = evidence.register_file(
        kind="table",
        description="Figure source data.",
        source_path=source,
        evidence_id="publication_figure_source_data",
    )
    metadata = {
        "figure_skill_policy_version": PUBLICATION_FIGURE_SKILL_POLICY_VERSION,
        "source_evidence_id": source_record.evidence_id,
        "source_evidence_ids": [source_record.evidence_id],
        "source_evidence_sha256": {
            source_record.evidence_id: source_record.sha256,
        },
    }
    contract = tmp_path / "figure_contract.json"
    contract.write_text("{}", encoding="utf-8")
    evidence.register_file(
        kind="log",
        description="Publication figure contract.",
        source_path=contract,
        evidence_id="publication_figure_contract",
        producer="publication_figure_skill",
        generation_mode="deterministic_figure_skill",
        metadata=metadata,
    )
    paths: dict[str, Path] = {}
    for suffix in ("svg", "png"):
        export = tmp_path / f"publication_figure.{suffix}"
        export.write_text("figure", encoding="utf-8")
        record = evidence.register_file(
            kind="figure",
            description="Publication figure export.",
            source_path=export,
            evidence_id=f"publication_figure_{suffix}",
            producer="publication_figure_skill",
            generation_mode="deterministic_figure_skill",
            metadata={"figure_role": "publication_figure", **metadata},
        )
        paths[suffix] = tmp_path / record.relative_path
    return evidence, paths


@pytest.mark.parametrize("mutation", ["delete", "tamper", "symlink"])
def test_publication_bundle_rejects_unverifiable_registered_export(
    tmp_path: Path,
    mutation: str,
) -> None:
    evidence, paths = _publication_bundle(tmp_path)
    target = paths["png"]
    if mutation == "delete":
        target.unlink()
    elif mutation == "tamper":
        target.write_text("different bytes", encoding="utf-8")
    else:
        target.unlink()
        outside = tmp_path / "outside.png"
        outside.write_text("figure", encoding="utf-8")
        try:
            target.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks unavailable")

    status = _publication_figure_bundle_ready(evidence=evidence, run_dir=tmp_path)

    assert status["publication_figure_bundle_ready"] is False
    assert status["publication_ready_stems"] == []


def test_unregistered_clean_audit_cannot_supersede_current_error(
    tmp_path: Path,
) -> None:
    evidence = EvidenceStore(tmp_path)
    path = (
        tmp_path
        / "evidence"
        / "publication_figure_skill_summary_v9__publication_figure_skill_summary.json"
    )
    path.write_text(json.dumps({"audit_findings": []}), encoding="utf-8")

    assert (
        _latest_publication_figure_audit_status(
            tmp_path,
            evidence=evidence,
            per_step_records=[],
        )
        is None
    )


def test_digest_bound_clean_audit_for_current_bundle_is_authoritative(
    tmp_path: Path,
) -> None:
    evidence, _paths = _publication_bundle(tmp_path)
    source_ids = [
        record.evidence_id
        for record in evidence.records()
        if record.evidence_id.startswith("publication_figure_source_")
    ]
    figure_ids = [
        record.evidence_id
        for record in evidence.records()
        if record.kind == "figure"
    ]
    evidence.register_json(
        kind="log",
        description="PublicationFigureSkill summary.",
        payload={
            "generated": True,
            "source_evidence_ids": source_ids,
            "figure_evidence_ids": figure_ids,
            "contract_evidence_id": "publication_figure_contract",
            "audit_findings": [],
        },
        filename="publication_figure_skill_summary.json",
        evidence_id="publication_figure_skill_summary",
    )

    status = _latest_publication_figure_audit_status(
        tmp_path,
        evidence=evidence,
        per_step_records=[],
    )

    assert status is not None
    assert status["error_count"] == 0


def test_clean_audit_loses_authority_when_its_source_step_fails(
    tmp_path: Path,
) -> None:
    evidence = EvidenceStore(tmp_path)
    source = tmp_path / "source.csv"
    source.write_text("term,estimate\nexposure,1.2\n", encoding="utf-8")
    source_record = evidence.register_file(
        kind="table",
        description="Current source.",
        source_path=source,
        evidence_id="current_source",
        produced_by_step="02_model",
    )
    evidence.register_json(
        kind="log",
        description="PublicationFigureSkill summary.",
        payload={
            "source_evidence_ids": [source_record.evidence_id],
            "audit_findings": [],
        },
        filename="publication_figure_skill_summary.json",
        evidence_id="publication_figure_skill_summary",
    )
    records = [
        {
            "step_id": "02_model",
            "status": "ok",
            "evidence_ids": [source_record.evidence_id],
        },
        {"step_id": "02_model", "status": "contract_failed"},
    ]

    assert (
        _latest_publication_figure_audit_status(
            tmp_path,
            evidence=evidence,
            per_step_records=records,
        )
        is None
    )


def test_tampered_clean_audit_cannot_supersede_current_error(tmp_path: Path) -> None:
    evidence = EvidenceStore(tmp_path)
    source = tmp_path / "source.csv"
    source.write_text("term,estimate\nexposure,1.2\n", encoding="utf-8")
    source_record = evidence.register_file(
        kind="table",
        description="Current source.",
        source_path=source,
        evidence_id="current_source",
    )
    summary_record = evidence.register_json(
        kind="log",
        description="PublicationFigureSkill summary.",
        payload={
            "source_evidence_ids": [source_record.evidence_id],
            "audit_findings": [],
        },
        filename="publication_figure_skill_summary.json",
        evidence_id="publication_figure_skill_summary",
    )
    (tmp_path / summary_record.relative_path).write_text(
        json.dumps({"source_evidence_ids": [source_record.evidence_id], "audit_findings": []}),
        encoding="utf-8",
    )

    assert (
        _latest_publication_figure_audit_status(
            tmp_path,
            evidence=evidence,
            per_step_records=[],
        )
        is None
    )


def test_invalid_new_audit_never_falls_back_to_older_clean_audit(
    tmp_path: Path,
) -> None:
    evidence, _paths = _publication_bundle(tmp_path)
    source_ids = [
        record.evidence_id
        for record in evidence.records()
        if record.evidence_id.startswith("publication_figure_source_")
    ]
    figure_ids = [
        record.evidence_id for record in evidence.records() if record.kind == "figure"
    ]
    payload = {
        "generated": True,
        "source_evidence_ids": source_ids,
        "figure_evidence_ids": figure_ids,
        "contract_evidence_id": "publication_figure_contract",
        "audit_findings": [],
    }
    evidence.register_json(
        kind="log",
        description="Older clean audit.",
        payload=payload,
        filename="publication_figure_skill_summary.json",
        evidence_id="publication_figure_skill_summary",
    )
    latest = evidence.register_json(
        kind="log",
        description="New clean audit.",
        payload=payload,
        filename="publication_figure_skill_summary.json",
        evidence_id="publication_figure_skill_summary_v2",
    )
    (tmp_path / latest.relative_path).write_text("{}", encoding="utf-8")

    assert (
        _latest_publication_figure_audit_status(
            tmp_path,
            evidence=evidence,
            per_step_records=[],
        )
        is None
    )
