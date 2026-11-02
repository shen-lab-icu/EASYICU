"""Equivalent review stops may share one preparation proposal; conflicts fail closed."""

import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.authority.plan_review import HumanReviewRequest, PlanReviewAuthority
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep
from easyicu.webserver import first_observed_preparation, raw_preparation_candidate as owner
from easyicu.webserver.pi_copilot import population_spec_review


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    plan = AnalysisPlan(research_question="Describe", steps=[AnalysisStep(step_id="describe", intent="Describe")])
    authority = PlanReviewAuthority.create(plan=plan)
    criterion = {"id": "c2", "kind": "first_observed_icu_single_hospital_adult", "role": "include"}
    population = {"plan_sha256": authority.plan_sha256, "criteria": [{"criterion": criterion, "disposition": "requires_extraction"}]}
    compiled = {"criteria": [{"criterion": criterion, "disposition": "requires_extraction"}]}
    def request(reason, **extra):
        payload = {"plan_review_authority": authority.model_dump(mode="json"), "reason": reason, **extra}
        return HumanReviewRequest.create(kind="scientific_stop", summary=reason,
            authority_sha256=canonical_sha256(payload), payload=payload)
    first = request("population", approval_allowed=False, population_compile=compiled,
                    population_compile_sha256=canonical_sha256(compiled))
    second = request("scientific", approval_allowed=True)
    pending = SimpleNamespace(run_dir=str(tmp_path), requests=[first, second])
    (tmp_path / "research_context.json").write_text("{}")
    monkeypatch.setattr(owner.agent_pipeline_runs, "raw_preparation_review_owner", lambda *args: (pending, {}))
    monkeypatch.setattr(owner, "parse_research_context_json", lambda _: object())
    monkeypatch.setattr(owner, "_constraints", lambda _: {"source_identity": {"database": "eicu_demo", "registration_id": "r"}})
    monkeypatch.setattr(population_spec_review, "reviewed_population_spec", lambda **_: population)
    monkeypatch.setattr(owner, "_first_observed_service_available", lambda: False)
    scope = {"criterion": criterion, "plan_sha256": authority.plan_sha256, "source_identity_sha256": "a" * 64,
             "purpose": "prepare_only", "analysis_authorized": False}
    seen = []
    def proposed(*, request, **kwargs):
        seen.append(request.review_id)
        return dict(scope)
    monkeypatch.setattr(first_observed_preparation, "proposed_first_observed_scope", proposed)
    return SimpleNamespace(pending=pending, authority=authority, first=first, second=second,
                           scope=scope, seen=seen, run=tmp_path, request=request)


def test_two_equivalent_stops_produce_one_preparation_without_analysis(candidate):
    result = owner.create_candidate_for_review(run_id="run", study_context_id="study")
    assert result["status"] == "pending" and result["analysis_authorized"] is False
    assert candidate.seen == [candidate.first.review_id, candidate.second.review_id]
    assert len(candidate.pending.requests) == 2
    stored = json.loads((candidate.run / "raw_preparation_checkpoint.json").read_text())
    assert stored["request"]["payload"]["first_observed_icu_preparation_scope"] == candidate.scope


@pytest.mark.parametrize("change", ["different_plan", "different_scope", "different_population", "bad_population_digest", "malformed_item"])
def test_non_equivalent_stops_refuse_to_issue_preparation(candidate, monkeypatch, change):
    if change == "different_plan":
        plan = AnalysisPlan(research_question="Other", steps=[AnalysisStep(step_id="describe", intent="Describe")])
        authority = PlanReviewAuthority.create(plan=plan)
        payload = {"plan_review_authority": authority.model_dump(mode="json")}
        candidate.pending.requests[1] = HumanReviewRequest.create(kind="scientific_stop", summary="other",
            authority_sha256=canonical_sha256(payload), payload=payload)
    elif change == "different_scope":
        monkeypatch.setattr(first_observed_preparation, "proposed_first_observed_scope",
            lambda *, request, **_: {**candidate.scope, "source_identity_sha256": "b" * 64}
            if request.review_id == candidate.second.review_id else dict(candidate.scope))
    else:
        compiled = {"criteria": [{"criterion": {"id": "c2", "kind": "age_years", "role": "include"},
                                  "disposition": "requires_extraction"}]}
        if change == "malformed_item":
            compiled["criteria"] = [{"criterion": candidate.scope["criterion"], "disposition": "requires_extraction"}, "invalid"]
        candidate.pending.requests[1] = candidate.request("different", approval_allowed=False,
            population_compile=compiled, population_compile_sha256=canonical_sha256(compiled) if change != "bad_population_digest" else "0" * 64)
    with pytest.raises(ValueError):
        owner.create_candidate_for_review(run_id="run", study_context_id="study")
    assert not (candidate.run / "raw_preparation_checkpoint.json").exists()
