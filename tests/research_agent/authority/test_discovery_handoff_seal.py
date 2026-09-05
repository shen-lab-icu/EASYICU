from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from easyicu.research_agent.discovery.discovery_handoff import (
    DiscoveryHandoffPacket,
    assert_discovery_analysis_ready,
    build_handoff_from_row,
    write_handoff_packet,
)


@pytest.fixture
def candidate(tmp_path):
    row = {
        "literature_idea_id": "candidate_1",
        "candidate_topic": "A clinical proposal",
        "go_no_go": "recommend",
        "go_no_go_reason": "feasibility reviewed",
        "resolved_outcome_concept": "death",
        "nested": {"evidence": ["pmid:123"]},
    }
    source = tmp_path / "triage.json"
    source.write_text(json.dumps({"discovery_ledger": [row]}))
    return row, source


def _packet(candidate, **kwargs):
    row, source = candidate
    return build_handoff_from_row(row, triage_report_path=source, **kwargs)


def test_freezes_both_fields_and_nested_values_without_aliasing(candidate):
    packet = _packet(candidate)
    candidate[0]["nested"]["evidence"].append("injected")
    assert packet.selected_ledger_row["nested"]["evidence"] == ("pmid:123",)
    with pytest.raises(ValidationError, match="frozen"):
        packet.database = "other"
    with pytest.raises(TypeError):
        packet.selected_ledger_row["nested"]["evidence"] = ()
    with pytest.raises(AttributeError):
        packet.inclusion_criteria.append("changed")
    wire = packet.model_dump(mode="json")
    wire["selected_ledger_row"]["nested"]["evidence"].append("changed")
    packet.verify_source()
    packet.model_copy(deep=True).verify_source()
    restored = DiscoveryHandoffPacket.model_validate_json(packet.model_dump_json())
    assert restored.handoff_sha256 == packet.handoff_sha256


@pytest.mark.parametrize(
    "field,value",
    [
        ("database", "eicu"),
        ("human_confirmed", True),
        ("research_question", "A different question"),
    ],
)
def test_unvalidated_copy_cannot_reuse_a_confirmation_seal(candidate, field, value):
    packet = _packet(candidate, human_confirmed=True)
    changed = packet.model_copy(
        update={field: value if field != "human_confirmed" else False}
    )
    with pytest.raises(ValueError, match="discovery_handoff_digest_mismatch"):
        changed.verify_seal()


def test_confirmation_and_source_bytes_are_both_required(candidate):
    packet = _packet(candidate, human_confirmed=True)
    assert assert_discovery_analysis_ready(packet)
    candidate[1].write_text("{}")
    assert packet.analysis_ready is False
    with pytest.raises(ValueError, match="discovery_source_evidence_changed"):
        assert_discovery_analysis_ready(packet)


def test_missing_source_cannot_be_sealed(candidate):
    candidate[1].unlink()
    with pytest.raises(ValueError, match="discovery_source_evidence_unavailable"):
        _packet(candidate, human_confirmed=True)


def test_wire_candidate_tampering_and_legacy_packet_are_refused(candidate):
    wire = _packet(candidate).model_dump(mode="json")
    wire["selected_ledger_row"]["go_no_go"] = "go"
    with pytest.raises(ValidationError, match="discovery_candidate_digest_mismatch"):
        DiscoveryHandoffPacket.model_validate(wire)
    wire["schema_version"] = "easyicu.discovery_handoff/3"
    with pytest.raises(ValidationError):
        DiscoveryHandoffPacket.model_validate(wire)


def test_new_confirmation_has_new_digest_and_cannot_overwrite_old_version(
    candidate, tmp_path
):
    proposal = _packet(candidate)
    confirmed = _packet(candidate, human_confirmed=True)
    assert confirmed.handoff_sha256 != proposal.handoff_sha256
    path = write_handoff_packet(proposal, tmp_path / "proposal.json")
    original = path.read_bytes()
    assert write_handoff_packet(proposal, path) == path
    with pytest.raises(ValueError, match="discovery_handoff_immutable_conflict"):
        write_handoff_packet(confirmed, path)
    assert path.read_bytes() == original
    write_handoff_packet(confirmed, tmp_path / "confirmation.json")


def test_plan_review_authority_freezes_the_complete_nested_plan():
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
    from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep

    plan = AnalysisPlan(
        research_question="A clinical question",
        steps=[
            AnalysisStep(
                step_id="summary",
                method="descriptive",
                intent="Summarize",
                inputs=[],
                expected_outputs=["table:summary"],
            ),
        ],
    )
    authority = PlanReviewAuthority.create(plan=plan)
    with pytest.raises(TypeError):
        authority.plan_payload["steps"][0]["method"] = "changed"
    with pytest.raises(TypeError):
        authority.evidence_sha256["new"] = "a" * 64
    wire = authority.model_dump(mode="json")
    wire["plan_payload"]["steps"][0]["method"] = "changed"
    with pytest.raises(ValidationError, match="plan_sha256"):
        PlanReviewAuthority.model_validate(wire)
