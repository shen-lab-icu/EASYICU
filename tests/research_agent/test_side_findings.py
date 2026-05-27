"""Side-finding archive and manuscript leak-guard tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def _finding_payload() -> dict:
    return {
        "finding_id": "side_lactate_distribution",
        "step_id": "descr_step_03",
        "title": "Lactate distribution note",
        "description": (
            "Lactate values showed a long right tail in the descriptive "
            "audit but this observation was not part of the primary analysis."
        ),
        "n": 42,
        "related_concept": "lactate",
    }


def test_coder_can_emit_side_findings() -> None:
    from easyicu.research_agent.side_findings import collect_side_findings

    findings = collect_side_findings(
        [
            {
                "step_id": "descr_step_03",
                "step_summary": {"side_findings": [_finding_payload()]},
            }
        ]
    )

    assert len(findings) == 1
    assert findings[0].finding_id == "side_lactate_distribution"
    assert findings[0].related_concept == "lactate"


def test_side_findings_md_generated_with_zero_count() -> None:
    from easyicu.research_agent.side_findings import render_side_findings_md

    text = render_side_findings_md([])

    assert "No side findings recorded." in text


def test_side_findings_md_structure() -> None:
    from easyicu.research_agent.side_findings import SideFinding, render_side_findings_md

    findings = [
        SideFinding.from_dict(_finding_payload()),
        SideFinding.from_dict(
            {
                **_finding_payload(),
                "finding_id": "side_map_distribution",
                "title": "MAP distribution note",
            }
        ),
    ]
    text = render_side_findings_md(findings)

    assert "## side_lactate_distribution (step=descr_step_03)" in text
    assert "## side_map_distribution (step=descr_step_03)" in text
    assert text.count("\n## ") == 2


def test_side_findings_excluded_from_writer_digest() -> None:
    from easyicu.research_agent.pipeline_writer_aux import (
        _render_writer_evidence_digest_v2,
    )

    digest = _render_writer_evidence_digest_v2(
        [
            {
                "step_id": "descr_step_03",
                "status": "ok",
                "step_summary": {"side_findings": [_finding_payload()]},
            }
        ]
    )

    assert "Lactate distribution note" not in digest
    assert "long right tail" not in digest


def test_strict_blocks_side_finding_leak_in_manuscript(ra) -> None:
    from easyicu.research_agent.manuscript_post import enforce_writer_claim_language
    from easyicu.research_agent.side_findings import SideFinding

    finding = SideFinding.from_dict(_finding_payload())
    with pytest.raises(ra.EvidenceEnforcementError) as exc_info:
        enforce_writer_claim_language(
            finding.description,
            enforcement_mode=ra.EvidenceEnforcementMode.STRICT,
            side_findings=[finding],
        )

    assert exc_info.value.detail["side_finding_leak"] == [finding.title]


def test_soft_annotates_side_finding_leak(ra) -> None:
    from easyicu.research_agent.manuscript_post import enforce_writer_claim_language
    from easyicu.research_agent.side_findings import SideFinding

    finding = SideFinding.from_dict(_finding_payload())
    annotated, detail = enforce_writer_claim_language(
        finding.description,
        enforcement_mode=ra.EvidenceEnforcementMode.SOFT,
        side_findings=[finding],
    )

    assert "<!-- SIDE_FINDING_LEAK:side_lactate_distribution -->" in annotated
    assert detail["side_finding_leak"] == [finding.title]


def test_manifest_records_side_findings_artifact(tmp_path: Path) -> None:
    from easyicu.research_agent.schema import AnalysisManifest

    manifest = AnalysisManifest(
        run_id="run",
        research_question="Question?",
        started_at="2026-05-27T00:00:00Z",
        context_path="context.json",
        side_findings_path="side_findings.md",
        side_findings_sha="abc",
        side_findings_count=2,
    )

    assert manifest.side_findings_sha == "abc"
    assert manifest.side_findings_count == 2


def test_side_findings_in_repro_envelope(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.side_findings import (
        SideFinding,
        write_side_findings,
    )

    evidence = ra.EvidenceStore(tmp_path)
    path, digest = write_side_findings(
        run_dir=tmp_path,
        findings=[SideFinding.from_dict(_finding_payload())],
        evidence=evidence,
        prompt_pack_version="test",
    )

    assert path.exists()
    assert digest
    record = evidence.get("side_findings")
    assert record is not None
    assert record.relative_path.endswith("side_findings.md")
