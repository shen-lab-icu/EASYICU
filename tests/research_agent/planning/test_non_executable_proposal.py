"""Non-executable drafts never grant analysis or turn unverified text into evidence."""
import pytest
from pydantic import ValidationError

from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.literature import CitationRecord, LiteratureBundle
from easyicu.research_agent.planning.method_literature import method_literature_citations
from easyicu.research_agent.planning.non_executable_proposal import (
    NonExecutableProposal, ProposedDecision, ProposedReference,
    build_non_executable_proposal, verify_proposal,
)


def _register(store, evidence_id, text):
    return store.register_text(kind="log", description="test", text=text,
                               filename=f"{evidence_id}.json", evidence_id=evidence_id)


def test_proposal_binds_exact_registered_evidence_and_keeps_missing_explicit(tmp_path):
    store = EvidenceStore(tmp_path)
    config = {"outcome": "case duration", "version": 1}
    citation = CitationRecord.model_validate(next(x for x in method_literature_citations() if x["key"] == "strobe_2007"))
    _register(store, "lit", LiteratureBundle(research_question="question", citations=[citation]).model_dump_json())
    _register(store, "source", '{"field":"duration"}')
    proposal = build_non_executable_proposal(store=store, configuration=config,
        expected_configuration_sha256=canonical_sha256(config), original_question="original question",
        decisions=(ProposedDecision(decision_id="methods", text="candidate model", references=(
            ProposedReference(evidence_id="lit", kind="literature", citation_key="strobe_2007"),)),
                   ProposedDecision(decision_id="outcome", text="candidate duration", references=(
            ProposedReference(evidence_id="source", kind="source"),))),
        unresolved=("source mapping unknown",))
    assert proposal.executable is proposal.review_approved is proposal.data_access_authorized is False
    assert proposal.decisions[0].unbound_evidence == ("source",)
    assert proposal.decisions[1].unbound_evidence == ("literature",)
    assert proposal.decisions[0].bindings[0].applicability_verified is False
    verify_proposal(store=store, proposal=proposal, configuration=config)
    with pytest.raises(ValueError, match="configuration_mismatch"):
        verify_proposal(store=store, proposal=proposal, configuration={"version": 2})
    source = store.get("source")
    (store.root / source.relative_path).write_text("tampered")
    with pytest.raises(ValueError, match="integrity_failed"):
        verify_proposal(store=store, proposal=proposal, configuration=config)


def test_unknown_evidence_and_unretained_citation_rejected(tmp_path):
    store = EvidenceStore(tmp_path)
    config = {"version": 1}
    _register(store, "lit", LiteratureBundle(research_question="q", citations=[]).model_dump_json())
    for ref, error in [(ProposedReference(evidence_id="missing", kind="source"), "not_registered"),
                       (ProposedReference(evidence_id="lit", kind="literature", citation_key="made-up"), "not_retained")]:
        with pytest.raises(ValueError, match=error):
            build_non_executable_proposal(store=store, configuration=config,
                expected_configuration_sha256=canonical_sha256(config), original_question="q",
                decisions=(ProposedDecision(decision_id="d", text="text", references=(ref,)),))


def test_missing_evidence_is_unbound_and_execution_cannot_be_granted(tmp_path):
    config = {}
    p = build_non_executable_proposal(store=EvidenceStore(tmp_path), configuration=config,
        expected_configuration_sha256=canonical_sha256(config), original_question="q",
        decisions=(ProposedDecision(decision_id="d", text="candidate"),), unresolved=("unknown",))
    assert p.decisions[0].unbound_evidence == ("source", "literature")
    assert p.decisions[0].evidence_status == "unbound"
    with pytest.raises(ValidationError):
        NonExecutableProposal.model_validate({**p.model_dump(), "executable": True})
