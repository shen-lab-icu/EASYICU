"""Evidence references must match retained source, without inventing appraisal."""
from datetime import datetime, timezone
from hashlib import sha256
from types import SimpleNamespace

from easyicu.research_agent.bibliographic_metadata import CitationRecord, LiteratureScreeningDecision
from easyicu.research_agent.literature import LiteratureBundle
from easyicu.research_agent.orchestration.scientific_plan_review_gate import append_required_literature_findings
from easyicu.research_agent.planning.literature_design_authority import (
    LiteratureDesignEvidence, LiteratureDesignEvidenceCard, _selected_source_fact,
)
from easyicu.research_agent.planning.literature_source_integrity import inspect_plan_literature_sources


EXCERPT = "The SICdb database contains records from an intensive care unit."


def bundle(*, excerpt=EXCERPT, digest=None, summary=EXCERPT, cards=True, role="analysis_method"):
    record = CitationRecord(key="source_1", title="SICdb: an openly available dataset", year="2023",
                            relevance="Study-design excerpt: " + excerpt)
    card = LiteratureDesignEvidenceCard(
        citation_key="source_1", evidence_role="database", access_mode="abstract",
        abstract_locator="https://pubmed.ncbi.nlm.nih.gov/37052626/",
        abstract_sha256=digest or sha256(EXCERPT.encode()).hexdigest(), supplement_status="unknown",
        reviewed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        evidence=[LiteratureDesignEvidence(dimension="study_population", review_role=role,
                                         source_backed_summary=summary, locator="Abstract")],
    )
    decision = LiteratureScreeningDecision(
        citation_key="source_1", source="pubmed", disposition="include", evidence_role="database",
        rationale="Candidate source resource for human review.",
    )
    return LiteratureBundle(research_question="A research question", citations=[record],
                            design_evidence_cards=[card] if cards else [], screening_decisions=[decision])


def plan():
    return SimpleNamespace(steps=[SimpleNamespace(literature_citation_keys=["source_1"])], design_selection=None)


def test_matching_abstract_is_only_integrity_checked():
    assert inspect_plan_literature_sources(plan(), bundle()) == ()


def test_changed_abstract_and_fabricated_fact_fail_closed():
    mismatch = inspect_plan_literature_sources(plan(), bundle(excerpt=EXCERPT + " Changed."))
    assert [(i.severity, i.reason) for i in mismatch] == [("error", "abstract_digest_mismatch")]
    fabricated = inspect_plan_literature_sources(plan(), bundle(summary="This paper supports an invented causal claim."))
    assert [(i.severity, i.reason) for i in fabricated] == [("error", "fact_not_in_source_excerpt")]


def test_missing_or_conflicting_source_is_not_silently_promoted():
    missing_card = inspect_plan_literature_sources(plan(), bundle(cards=False))
    assert [(i.severity, i.reason) for i in missing_card] == [("warning", "no_source_evidence_card")]
    b = bundle()
    b.screening_decisions.append(b.screening_decisions[0].model_copy(update={"disposition": "exclude"}))
    assert inspect_plan_literature_sources(plan(), b)[0].reason == "source_not_included"


def test_literature_integrity_is_projected_into_review_findings():
    findings = []
    config = SimpleNamespace(require_literature_design_authority=False, require_literature_retrieval_evidence=False)
    append_required_literature_findings(config, findings, plan(), bundle(excerpt=EXCERPT + " Changed."))
    assert len(findings) == 1 and findings[0].severity == "error"
    assert findings[0].detail["sources"][0]["reason"] == "abstract_digest_mismatch"
    assert findings[0].detail["excerpt_match_does_not_prove_applicability"] is True


def test_source_using_study_is_not_promoted_to_database_resource():
    application = "This multicenter prediction study used the SICdb database."
    assert _selected_source_fact([application], "sic", title="ICU mortality prediction using SICdb") == ""
    resource = "The SICdb database contains records from an intensive care unit."
    assert _selected_source_fact([resource], "sic", title="SICdb: an openly available dataset") == resource
    assert _selected_source_fact([resource], "miiv", title="SICdb: an openly available dataset") == ""


def test_historical_application_source_role_is_flagged_without_rewriting_bundle():
    old = bundle(role="data_source")
    old.citations[0] = old.citations[0].model_copy(update={"title": "ICU mortality prediction using SICdb"})
    before = old.model_dump_json()
    issues = inspect_plan_literature_sources(plan(), old)
    assert [(i.severity, i.reason) for i in issues] == [
        ("warning", "database_role_not_resource_description"),
        ("warning", "source_resource_identity_unverified"),
    ]
    assert old.model_dump_json() == before


def _real_plan():
    from tests.research_agent.planning.scientific_review_fixtures import _plan
    from easyicu.research_agent.planning.literature_contract import LiteratureDesignBinding
    original = _plan()
    steps = []
    for step in original.steps:
        if step.literature_citation_keys:
            step = step.model_copy(update={
                "literature_citation_keys": ["source_1"],
                "literature_design_bindings": [LiteratureDesignBinding(
                    citation_key="source_1", design_elements=["population"],
                    application="Describe the population of the original database resource.",
                )],
            })
        steps.append(step)
    return original.model_copy(update={"steps": steps})


def test_digest_and_fact_integrity_block_the_actual_persisted_review_gate(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
    from easyicu.research_agent.orchestration.scientific_plan_review_gate import prepare_scientific_plan_review_gate
    from easyicu.webserver.agent_pipeline_runs import _load_pending_scientific_review, _pending_plan_approval_allowed
    from tests.research_agent.planning.scientific_review_fixtures import _context

    conflicting = bundle()
    conflicting.screening_decisions.append(
        conflicting.screening_decisions[0].model_copy(update={"disposition": "exclude"})
    )
    for index, source in enumerate([
        bundle(excerpt=EXCERPT + " Changed."),
        bundle(summary="This paper supports an invented causal claim."),
        conflicting,
        # Merely omitting the source screening decision cannot grant curated
        # authority, even if its retained excerpt and card are consistent.
        bundle().model_copy(update={"screening_decisions": []}),
    ]):
        run_dir = tmp_path / str(index)
        run_dir.mkdir()
        gate = prepare_scientific_plan_review_gate(
            context=_context(), plan=_real_plan(), literature=source, figure_strategy=None,
            run_dir=run_dir, evidence=EvidenceStore(run_dir), model_retention=None,
        )
        assert not gate.review.approval_allowed
        assert gate.review.status == "changes_required"
        assert any(f.code == "LITERATURE_SOURCE_INTEGRITY_INVALID" and f.severity == "blocker"
                   for f in gate.review.findings)
        assert gate.finding.detail["approval_allowed"] is False
        assert '"LITERATURE_SOURCE_INTEGRITY_INVALID"' in gate.artifact_path.read_text()
        authority = PlanReviewAuthority.create(
            plan=_real_plan(),
            evidence_sha256={"scientific_plan_review": sha256(gate.artifact_path.read_bytes()).hexdigest()},
        )
        pending = SimpleNamespace(requests=[SimpleNamespace(
            payload={"plan_review_authority": authority.model_dump(mode="json")},
        )])
        assert _load_pending_scientific_review(run_dir, pending)["approval_allowed"] is False
        assert not _pending_plan_approval_allowed(
            run_dir=run_dir, pending=pending, plan_recommendation_complete=True,
        )


def test_unavailable_source_remains_explicit_without_becoming_integrity_blocker():
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    from tests.research_agent.planning.scientific_review_fixtures import _context
    review = build_plan_scientific_review(context=_context(), plan=_real_plan(), literature=bundle(cards=False))
    issue = next(f for f in review.findings if f.code == "LITERATURE_SOURCE_EVIDENCE_UNVERIFIED")
    assert issue.severity == "major"
    assert "no_source_evidence_card" in issue.message
    assert not review.top_journal_candidate


def test_older_review_schema_cannot_authorize_a_new_plan(tmp_path):
    from easyicu.research_agent.authority.plan_review import PlanReviewAuthority
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    from easyicu.webserver.agent_pipeline_runs import _load_pending_scientific_review
    from tests.research_agent.planning.scientific_review_fixtures import _context

    review = build_plan_scientific_review(context=_context(), plan=_real_plan(), literature=bundle())
    # /20 predates the screening-and-curated-identity contract.  Even an
    # otherwise matching, previously approved record must be re-reviewed.
    archived = review.model_copy(update={"schema_version": "easyicu.plan_scientific_review/20"})
    path = tmp_path / "scientific_plan_review.json"
    path.write_text(archived.model_dump_json(), encoding="utf-8")
    authority = PlanReviewAuthority.create(
        plan=_real_plan(), evidence_sha256={"scientific_plan_review": sha256(path.read_bytes()).hexdigest()},
    )
    pending = SimpleNamespace(requests=[SimpleNamespace(
        payload={"plan_review_authority": authority.model_dump(mode="json")},
    )])
    assert _load_pending_scientific_review(tmp_path, pending) == {}


def test_old_review_cannot_be_reused_after_citation_authority_change(tmp_path):
    import pytest

    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.orchestration.scientific_plan_review_gate import (
        ScientificPlanReviewArtifactError, persist_or_validate_scientific_plan_review,
    )
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    from tests.research_agent.planning.scientific_review_fixtures import _context

    current = build_plan_scientific_review(context=_context(), plan=_real_plan(), literature=bundle())
    archived = current.model_copy(update={"schema_version": "easyicu.plan_scientific_review/20"})
    evidence = EvidenceStore(tmp_path)
    persist_or_validate_scientific_plan_review(run_dir=tmp_path, evidence=evidence, current_review=archived)
    with pytest.raises(ScientificPlanReviewArtifactError) as caught:
        persist_or_validate_scientific_plan_review(
            run_dir=tmp_path, evidence=evidence, current_review=current, reuse_existing_review=True,
        )
    assert caught.value.code == "scientific_plan_review_binding_drift"
