"""Reviewed population criteria reach the existing extraction contract exactly."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from easyicu.research_agent.orchestration.progressive_planning import (
    population_approval_findings,
    population_proposals_finding,
)
from easyicu.research_agent.orchestration.workflow import human_review_requests_for_plan
from easyicu.research_agent.planning.population_compile import compile_population
from easyicu.research_agent.planning.population_spec import PopulationSpec
from easyicu.research_agent.schema import (
    AnalysisPlan,
    CohortDescriptor,
    ResearchContext,
    UserPreferences,
)
from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot import extraction_handoff


def reviewed_population(*criteria, time_zero=None, variables=(), question=None, settings=None):
    spec = PopulationSpec.model_validate(
        {
            "criteria": [
                {
                    "id": f"c{i}",
                    "quote": f"Criterion {i}",
                    "source": "question",
                    "role": "include",
                    **row,
                }
                for i, row in enumerate(criteria, 1)
            ]
        }
    )
    context = ResearchContext(
        research_question=question or "Describe the stated population",
        cohort=CohortDescriptor(cohort_name="synthetic", database="miiv", n_stays=0),
        variables=list(variables),
        user_preferences=UserPreferences(data_constraints=json.dumps({"cohort": settings or {}})),
    )
    population = compile_population(spec, context, time_zero_hours=time_zero)
    plan = AnalysisPlan(
        research_question=context.research_question,
        steps=[],
        cohort=population.cohort_definition(),
    )
    requests = human_review_requests_for_plan(
        findings=population_approval_findings(population),
        plan=plan,
        require_plan_review=True,
    )
    return population, requests


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setenv("EASYICU_HOME", str(tmp_path))
    return study_contexts.upsert_context(
        {
            "question": "Describe first ICU stays",
            "cohort": {"preset": "all_icu", "max_patients": 57},
            "data_source": {"database": "miiv", "path": "/synthetic/prepared"},
            "modules": ["demographics"],
        }
    )


def proposal(study, requests):
    from easyicu.webserver.pi_copilot.population_preparation import (
        population_preparation_proposal,
    )

    return population_preparation_proposal(
        study=study, run_id="run_population", requests=requests
    )


def test_review_keeps_the_complete_compilation_bound_to_its_plan(study):
    population, requests = reviewed_population({"kind": "first_icu_stay"})
    assert requests[0].payload.get("population_compile") == population.record()
    assert requests[0].payload["population_compile_sha256"] == population.sha256()
    assert requests[0].payload["approval_allowed"] is False


def test_a_lifecycle_handoff_refreshes_the_proposal_but_not_the_old_click(
    study, service_fixture
):
    from easyicu.webserver import agent_pipeline_runs
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError

    service, record, old_body, _, _ = service_fixture
    current = study_contexts.handoff_context(study["id"], current_stage="agent_planned")
    record.binding = record.binding.model_copy(
        update={"study_revision": current["revision"]}
    )
    review = agent_pipeline_runs.pending_review(old_body["run_id"])
    proposal = review["population_preparation"]
    assert proposal["study_revision"] == current["revision"]
    with pytest.raises(PiCopilotError):
        service.apply_population_preparation(record.session_id, **old_body)
    service.apply_population_preparation(
        record.session_id,
        **{
            **old_body,
            "expected_revision": current["revision"],
            "proposal_sha256": proposal["proposal_sha256"],
        },
    )
    assert (
        study_contexts.get_context(study["id"])["cohort"]["exclude_readmissions"]
        is True
    )


def test_an_explicit_projection_reaches_extraction_without_starting_it(study):
    _, requests = reviewed_population({"kind": "first_icu_stay", "source": "outline"})
    result = proposal(study, requests)
    assert result["available"] is True
    assert result["criteria"][0]["proposed_by_system"] is True
    assert study_contexts.get_context(study["id"])["cohort"] == study["cohort"]
    updated = study_contexts.upsert_context(
        {"id": study["id"], "cohort": result["cohort"]},
        expected_revision=study["revision"],
        require_revision=True,
        lifecycle_write=False,
    )
    calls = []
    extraction_handoff.submit_study_extraction(
        study=updated,
        registered_source=None,
        submit=lambda body: calls.append(body) or {"job_id": "synthetic"},
    )
    assert calls[0]["cohort"]["exclude_readmissions"] is True
    assert calls[0]["cohort"]["max_patients"] == 57
    assert calls[0]["study_context_revision"] == updated["revision"]
    assert calls[0]["path"] == "/synthetic/prepared"


@pytest.mark.parametrize("source", ["outline", "preset"])
@pytest.mark.parametrize("criterion,question,settings", [
    (
        {"kind": "icu_stay_hours", "min_hours": 24, "quote": "first 24 hours"},
        "Describe maximum lactate during the first 24 hours of the ICU stay.",
        {"min_icu_los_hours": 24, "label": "Adults with sepsis"},
    ),
    (
        {"kind": "age_years", "min_years": 18, "max_years": 89, "quote": "adults"},
        "Describe outcomes in adults.",
        {"age_min": 18, "age_max": 89, "label": "Adults with sepsis"},
    ),
])
def test_preparation_and_system_proposals_keep_the_compilers_attribution(
    study, source, criterion, question, settings,
):
    population, requests = reviewed_population(
        {**criterion, "source": source}, question=question, settings=settings,
    )
    result = proposal(study, requests)
    proposed = population_proposals_finding(population).detail["criteria"][0]
    row = result["criteria"][0]
    assert row["source"] == source
    assert row["proposed_by_system"] is True
    assert row["stated_by_study"] is False
    assert row["quote"] == criterion["quote"]
    assert row["quote_found_in_question"] is True
    assert row["attribution"] == "typed_study_setting"
    assert row["typed_study_setting"]
    for field in ("source", "stated_by_study", "quote_found_in_question", "attribution", "typed_study_setting"):
        assert row[field] == proposed[field]


def test_omission_card_keeps_system_interpretation_separate_from_the_question(study):
    from easyicu.webserver.pi_copilot.population_acceptance import population_acceptance_proposal

    population, requests = reviewed_population(
        {"kind": "not_typed", "why": "No supported restriction kind", "source": "outline", "quote": "adults"},
        question="Describe outcomes in adults.",
    )
    result = population_acceptance_proposal(
        study=study, pending=SimpleNamespace(requests=requests, run_id="run_population"),
    )
    row = result["criteria"][0]
    assert row["proposed_by_system"] is True
    assert row["stated_by_study"] is False
    assert row["quote_found_in_question"] is True
    assert row["attribution"] == "outline"
    assert population.criteria[0].disposition == "not_applied"


@pytest.mark.parametrize(
    "criterion,field,value",
    [
        ({"kind": "age_years", "min_years": 21}, "age_min", 21),
        ({"kind": "age_years", "max_years": 75}, "age_max", 75),
        ({"kind": "icu_stay_hours", "min_hours": 12}, "min_icu_los_hours", 12),
    ],
)
def test_supported_bounds_are_preserved(study, criterion, field, value):
    _, requests = reviewed_population(criterion, time_zero=24)
    result = proposal(study, requests)
    assert result["available"] is True
    assert (
        extraction_handoff.compile_study_cohort({**study, "cohort": result["cohort"]})[
            field
        ]
        == value
    )


@pytest.mark.parametrize(
    "criterion",
    [
        {"kind": "age_years", "min_years": 18.5},
        {"kind": "age_years", "max_years": 125},
        {"kind": "age_years", "min_years": 0},
        {"kind": "age_years", "min_years": 101},
        {"kind": "age_years", "max_years": 100},
        {"kind": "icu_stay_hours", "min_hours": 36},
        {"kind": "icu_stay_hours", "min_hours": 12, "max_hours": 24},
        {"kind": "diagnosis_codes", "system": "icd10", "codes": ["I50"]},
        {"kind": "condition_present", "concepts_all_of": ["sep3"], "window": None},
    ],
)
def test_unrepresentable_criteria_never_become_approximate_extractions(
    study, criterion
):
    _, requests = reviewed_population(criterion, time_zero=24)
    result = proposal(study, requests)
    assert result is not None and result["available"] is False
    assert "cohort" not in result


def test_a_first_stay_proposal_cannot_override_explicit_all_stays(study):
    _, requests = reviewed_population({"kind": "first_icu_stay"})
    result = proposal({**study, "cohort": {"exclude_readmissions": False}}, requests)
    assert result["available"] is False


@pytest.mark.parametrize("damage", ["missing", "digest", "plan", "duplicate"])
def test_incomplete_or_drifted_review_material_offers_no_action(study, damage):
    _, requests = reviewed_population({"kind": "first_icu_stay"})
    payload = deepcopy(dict(requests[0].payload))
    if damage == "missing":
        payload.pop("population_compile", None)
    elif damage == "digest":
        payload["population_compile_sha256"] = "0" * 64
    elif damage == "plan":
        payload["plan_review_authority"]["plan_sha256"] = "0" * 64
    changed = [
        SimpleNamespace(payload=payload, authority_sha256=requests[0].authority_sha256)
    ]
    if damage == "duplicate":
        changed *= 2
    assert proposal(study, changed) is None


def test_other_unapplied_inclusions_keep_the_whole_preparation_blocked(study):
    _, requests = reviewed_population(
        {"kind": "first_icu_stay"},
        {"kind": "not_typed", "why": "No exact supported kind"},
    )
    result = proposal(study, requests)
    assert result["available"] is False
    assert len(result["criteria"]) == 2


@pytest.fixture
def service_fixture(study, tmp_path, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs
    from easyicu.webserver.pi_copilot.service import PiCopilotService
    from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiSessionRecord

    _, requests = reviewed_population({"kind": "first_icu_stay"})
    run_id = "run_20261015T120000_abc123"
    pending = SimpleNamespace(
        run_id=run_id,
        run_dir=str(tmp_path),
        requests=requests,
        resume_scope="plan",
        resumable_here=True,
    )
    entry = SimpleNamespace(
        pending=pending,
        study=study,
        credential_source="synthetic",
        budget_mode="full_reviewed",
        provider={},
        wrapper_dir=tmp_path,
    )
    monkeypatch.setattr(agent_pipeline_runs, "_PENDING_REVIEWS", {run_id: entry})
    review = agent_pipeline_runs.pending_review(run_id)
    assert review["population_preparation"]["available"] is True
    service = PiCopilotService.__new__(PiCopilotService)
    record = PiSessionRecord(
        session_id="pi-population",
        project_id="project",
        binding=AuthorityBinding(
            study_context_id=study["id"],
            study_revision=study["revision"],
            run_id=run_id,
        ),
    )
    saved = []
    monkeypatch.setattr(service, "_scoped_record", lambda *a, **kw: record)
    monkeypatch.setattr(service, "_stale_details", lambda *a: {})
    monkeypatch.setattr(service, "_save_record", lambda item: saved.append(item))
    body = {
        "project_id": "project",
        "expected_revision": study["revision"],
        "run_id": run_id,
        "proposal_sha256": review["population_preparation"]["proposal_sha256"],
    }
    return service, record, body, review, saved


def test_public_pending_review_to_service_to_extraction(study, service_fixture):
    from easyicu.webserver.pi_copilot import plan_review_progress
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError

    service, record, body, review, saved = service_fixture
    result = service.apply_population_preparation(record.session_id, **body)
    updated = study_contexts.get_context(study["id"])
    assert result["next_action"] == "request_extraction"
    assert updated["revision"] == study["revision"] + 1
    assert updated["cohort"]["exclude_readmissions"] is True
    assert updated.get("active_job_id") is None
    assert record.binding.study_revision == updated["revision"]
    assert len(saved) == 1
    receipt = plan_review_progress.matching_progress(updated, review)
    assert receipt.choices["population_preparation"] == body["proposal_sha256"]
    calls = []
    extraction_handoff.submit_study_extraction(
        study=updated,
        registered_source=None,
        submit=lambda payload: calls.append(payload) or {"job_id": "synthetic"},
    )
    assert calls[0]["cohort"]["exclude_readmissions"] is True
    assert calls[0]["cohort"]["max_patients"] == 57
    with pytest.raises(PiCopilotError):
        service.apply_population_preparation(record.session_id, **body)
    assert study_contexts.get_context(study["id"])["revision"] == updated["revision"]


@pytest.mark.parametrize(
    "change", ["revision", "run", "digest", "question", "active_job", "missing_review"]
)
def test_a_stale_or_busy_action_writes_nothing(
    study, service_fixture, monkeypatch, change
):
    from easyicu.webserver import agent_pipeline_runs
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError

    service, record, body, _, saved = service_fixture
    if change == "revision":
        body["expected_revision"] += 1
    elif change == "run":
        body["run_id"] = "run_other"
    elif change == "digest":
        body["proposal_sha256"] = "0" * 64
    elif change == "missing_review":
        monkeypatch.setattr(agent_pipeline_runs, "pending_review", lambda *a: None)
    else:
        changed = (
            {**study, "question": "another study"}
            if change == "question"
            else {**study, "active_job_id": "running"}
        )
        monkeypatch.setattr(study_contexts, "get_context", lambda *a: changed)
    with pytest.raises(PiCopilotError):
        service.apply_population_preparation(record.session_id, **body)
    assert saved == []
    assert study_contexts.get_context(study["id"])["cohort"] == study["cohort"]


def test_a_concurrent_revision_wins_before_the_population_write(
    study, service_fixture, monkeypatch
):
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError

    service, record, body, _, saved = service_fixture
    real = study_contexts.upsert_context

    def raced(payload, **kwargs):
        real({"id": study["id"], "question": "A concurrently edited question"})
        return real(payload, **kwargs)

    monkeypatch.setattr(study_contexts, "upsert_context", raced)
    with pytest.raises(PiCopilotError, match="changed before saving"):
        service.apply_population_preparation(record.session_id, **body)
    assert study_contexts.get_context(study["id"])["cohort"] == study["cohort"]
    assert saved == []


def test_the_route_accepts_only_the_displayed_proposal_coordinates(
    service_fixture, monkeypatch
):
    from fastapi.testclient import TestClient
    from easyicu.webserver.app import app
    from easyicu.webserver.routes import pi_copilot

    service, record, body, _, _ = service_fixture
    monkeypatch.setattr(pi_copilot, "get_pi_copilot_service", lambda: service)
    client = TestClient(app)
    url = f"/api/copilot/pi/sessions/{record.session_id}/population-preparation"
    for field in ("cohort", "grants", "prompt"):
        assert client.post(url, json={**body, field: {}}).status_code == 422
    result = client.post(url, json=body)
    assert result.status_code == 200, result.text
    assert result.json()["next_action"] == "request_extraction"


def test_a_changed_cohort_cannot_reuse_the_old_proposal(study):
    from easyicu.webserver.pi_copilot.population_preparation import (
        current_population_preparation,
    )

    _, requests = reviewed_population({"kind": "first_icu_stay"})
    result = proposal(study, requests)
    review = {"run_id": result["run_id"], "population_preparation": result}
    assert current_population_preparation(study, review) == result
    assert (
        current_population_preparation(
            {**study, "revision": study["revision"] + 1}, review
        )
        is None
    )
    assert (
        current_population_preparation(
            {**study, "question": "A different question"}, review
        )
        is None
    )


def test_an_existing_stricter_restriction_is_not_widened(study):
    _, requests = reviewed_population(
        {"kind": "age_years", "min_years": 18}, {"kind": "first_icu_stay"}
    )
    result = proposal({**study, "cohort": {**study["cohort"], "age_min": 25}}, requests)
    assert result["cohort"]["age_min"] == 25
    assert result["changes"] == [
        {"field": "exclude_readmissions", "before": False, "after": True}
    ]


def test_an_already_configured_extraction_needs_no_repeated_setting_change(study):
    _, requests = reviewed_population({"kind": "first_icu_stay"})
    result = proposal({**study, "cohort": {"exclude_readmissions": True}}, requests)
    assert result["available"] is False
    assert result["status"] == "already_configured"


def test_an_applied_plan_predicate_is_kept_beside_the_extraction_remedy(study):
    from easyicu.research_agent.schema import ConceptDescriptor, VariableRole

    population, requests = reviewed_population(
        {"kind": "age_years", "min_years": 18},
        {"kind": "first_icu_stay"},
        variables=[
            ConceptDescriptor(
                name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64", unit="years"
            )
        ],
    )
    assert population.criteria[0].disposition == "applied_by_plan"
    result = proposal(study, requests)
    assert result["available"] is True
    assert result["criteria"][0]["disposition"] == "applied_by_plan"
    assert "age_min" not in result["cohort"]


def test_a_plan_reshaped_after_compilation_does_not_offer_the_old_population(study):
    from easyicu.research_agent.canonical_json import canonical_sha256

    _, requests = reviewed_population({"kind": "first_icu_stay"})
    payload = deepcopy(requests[0].payload)
    plan_authority = payload["plan_review_authority"]
    plan_authority["plan_payload"]["cohort"]["unapplied_population_criteria"] = []
    plan_authority["plan_sha256"] = canonical_sha256(plan_authority["plan_payload"])
    request = SimpleNamespace(
        payload=payload, authority_sha256=canonical_sha256(payload)
    )
    assert proposal(study, [request]) is None


def test_the_workflow_projects_the_bound_proposal_and_drops_it_after_an_edit(
    study, service_fixture
):
    from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot

    _, _, body, review, _ = service_fixture
    review = {**review, "scientific_plan_review": {"status": "changes_required"}}
    run = {
        "run_id": body["run_id"],
        "run_type": "full",
        "run_status": "human_review_pending",
        "artifact_names": ["agent_plan.json"],
        "scientific_configuration_sha256": review["scientific_configuration_sha256"],
    }

    def snapshot(current):
        return build_research_workflow_snapshot(
            study=current,
            active_export_present=True,
            active_job=None,
            latest_run=run,
            plan_review_authority=review,
        )

    assert (
        snapshot(study).plan_review_summary["population_preparation"]["available"]
        is True
    )
    assert (
        "population_preparation"
        not in snapshot({**study, "question": "Edited question"}).plan_review_summary
    )


@pytest.fixture
def acceptance_fixture(study, service_fixture, monkeypatch):
    from easyicu.webserver import agent_pipeline_runs

    service, record, body, _, saved = service_fixture
    _, requests = reviewed_population(
        {"kind": "not_typed", "why": "No exact diagnosis-state representation"},
        {"kind": "not_typed", "why": "No exact prior-treatment representation"},
    )
    entry = agent_pipeline_runs._PENDING_REVIEWS[body["run_id"]]
    entry.pending.requests = requests
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_plan_has_complete_reviewable_recommendation",
        lambda *_: True,
    )
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_load_pending_scientific_review",
        lambda *_: {
            "schema_version": agent_pipeline_runs.CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION,
            "approval_allowed": True,
            "findings": [],
        },
    )
    return service, record, body["run_id"], entry, saved


def accept_current(service, record, run_id, criterion_id):
    from easyicu.webserver import agent_pipeline_runs

    proposal = agent_pipeline_runs.pending_review(run_id)["population_acceptance"]
    return service.accept_population_inclusion(
        record.session_id,
        project_id=record.project_id,
        expected_revision=record.binding.study_revision,
        run_id=run_id,
        proposal_sha256=proposal["proposal_sha256"],
        criterion_id=criterion_id,
    )


def test_two_choices_enable_full_plan_review_without_starting_it(
    study, acceptance_fixture
):
    from easyicu.webserver import agent_pipeline_runs

    service, record, run_id, entry, saved = acceptance_fixture
    assert agent_pipeline_runs.pending_review(run_id)["plan_approval_allowed"] is False
    accept_current(service, record, run_id, "c1")
    review = agent_pipeline_runs.pending_review(run_id)
    assert review["plan_approval_allowed"] is False
    assert review["population_acceptance"]["accepted_criterion_ids"] == ["c1"]
    accept_current(service, record, run_id, "c2")
    review = agent_pipeline_runs.pending_review(run_id)
    assert review["plan_approval_allowed"] is True
    assert review["requests"][0]["reason_code"] == "operator_plan_approval_required"
    assert review["population_acceptance"]["complete"] is True
    assert entry.pending.requests[0].payload["approval_allowed"] is False
    assert study_contexts.get_context(study["id"]) == study
    assert saved == []


def test_copilot_keeps_accepted_omissions_visible_at_the_full_plan_review(
    study, acceptance_fixture
):
    from easyicu.webserver import agent_pipeline_runs
    from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot

    service, record, run_id, _, _ = acceptance_fixture

    def snapshot(current):
        review = agent_pipeline_runs.pending_review(run_id)
        return build_research_workflow_snapshot(
            study=current,
            active_export_present=True,
            active_job=None,
            latest_run={
                "run_id": run_id,
                "run_type": "full",
                "run_status": "human_review_pending",
                "artifact_names": ["agent_plan.json"],
                "scientific_configuration_sha256": review[
                    "scientific_configuration_sha256"
                ],
            },
            plan_review_authority=review,
        )

    assert snapshot(study).next_action_code == "population_inclusion_not_applied"
    accept_current(service, record, run_id, "c1")
    assert snapshot(study).next_action_code == "population_inclusion_not_applied"
    accept_current(service, record, run_id, "c2")
    final = snapshot(study)
    assert final.next_action_code == "operator_plan_approval_required"
    assert final.plan_review_summary["population_acceptance"]["complete"] is True
    assert (
        "population_acceptance"
        not in snapshot({**study, "revision": study["revision"] + 1}).plan_review_summary
    )


@pytest.mark.parametrize("identifier", [None, "", "../invalid"])
def test_unbound_historical_study_has_no_current_population_proposal(identifier):
    from easyicu.webserver.pi_copilot.population_acceptance import current_review_study

    assert current_review_study({"id": identifier}) is None


@pytest.mark.parametrize(
    "change",
    ["revision", "run", "digest", "criterion", "study", "job", "concurrent_choice"],
)
def test_omission_choices_refuse_stale_or_wrong_coordinates(
    study, acceptance_fixture, monkeypatch, change
):
    from easyicu.webserver import agent_pipeline_runs
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError

    service, record, run_id, _, saved = acceptance_fixture
    proposal = agent_pipeline_runs.pending_review(run_id)["population_acceptance"]
    body = dict(
        project_id=record.project_id,
        expected_revision=record.binding.study_revision,
        run_id=run_id,
        proposal_sha256=proposal["proposal_sha256"],
        criterion_id="c1",
    )
    if change == "revision":
        body["expected_revision"] += 1
    elif change == "run":
        body["run_id"] = "other"
    elif change == "digest":
        body["proposal_sha256"] = "a" * 64
    elif change == "criterion":
        body["criterion_id"] = "c99"
    elif change == "study":
        study_contexts.upsert_context(
            {"id": study["id"], "question": "Changed scientific scope"}
        )
    elif change == "job":
        study_contexts.handoff_context(study["id"], active_job_id="running")
    else:
        accept_current(service, record, run_id, "c2")
    with pytest.raises(PiCopilotError):
        service.accept_population_inclusion(record.session_id, **body)
    assert saved == []
    assert study_contexts.get_context(study["id"])["cohort"] == study["cohort"]


def test_accepted_omissions_do_not_remove_other_scientific_requirements(
    acceptance_fixture, monkeypatch
):
    from easyicu.webserver import agent_pipeline_runs

    service, record, run_id, entry, _ = acceptance_fixture
    accept_current(service, record, run_id, "c1")
    accept_current(service, record, run_id, "c2")
    monkeypatch.setattr(
        agent_pipeline_runs,
        "_load_pending_scientific_review",
        lambda *_: {
            "schema_version": agent_pipeline_runs.CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION,
            "approval_allowed": False,
            "findings": [],
        },
    )
    assert agent_pipeline_runs.pending_review(run_id)["plan_approval_allowed"] is False
    # Changing the offered plan/request also changes the acceptance coordinates.
    _, entry.pending.requests = reviewed_population({"kind": "first_icu_stay"})
    review = agent_pipeline_runs.pending_review(run_id)
    assert review["population_acceptance"] is None
    assert review["plan_approval_allowed"] is False


def test_corrupt_acceptance_receipt_is_not_silently_replaced(
    acceptance_fixture, tmp_path
):
    from easyicu.webserver import agent_pipeline_runs
    from easyicu.webserver.pi_copilot.contracts import PiCopilotError

    service, record, run_id, _, _ = acceptance_fixture
    accept_current(service, record, run_id, "c1")
    (receipt,) = tmp_path.glob("**/population-acceptances/*.json")
    receipt.write_text('{"choices":{"c1":{},"c2":{}}}')
    review = agent_pipeline_runs.pending_review(run_id)
    assert review["population_acceptance"]["record_invalid"] is True
    assert review["plan_approval_allowed"] is False
    with pytest.raises(PiCopilotError):
        accept_current(service, record, run_id, "c2")
    assert receipt.read_text() == '{"choices":{"c1":{},"c2":{}}}'


def test_final_host_resume_carries_choices_from_before_job_handoff(
    study, acceptance_fixture, monkeypatch
):
    from easyicu.webserver import agent_pipeline_runs

    service, record, run_id, entry, _ = acceptance_fixture
    accept_current(service, record, run_id, "c1")
    accept_current(service, record, run_id, "c2")
    captured = study_contexts.get_context(study["id"])
    study_contexts.handoff_context(study["id"], active_job_id="approved-job")
    entry.pipeline = SimpleNamespace()
    entry.prepared_package_binding = {"synthetic": True}
    monkeypatch.setattr(
        agent_pipeline_runs.dataio,
        "validate_research_pipeline_source",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        agent_pipeline_runs.run_patient_grouping,
        "require_patient_groups_for_approval",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(agent_pipeline_runs, "_progress", lambda *a, **kw: None)
    decisions = []

    class BoundaryReached(Exception):
        pass

    def capture(_registry, _entry, rows, **kwargs):
        decisions.extend(rows)
        raise BoundaryReached

    monkeypatch.setattr(agent_pipeline_runs, "resume_pending_review", capture)
    with pytest.raises(BoundaryReached):
        agent_pipeline_runs.resume_research_pipeline(
            run_id=run_id,
            study_context_id=study["id"],
            decision="approved",
            reviewer="test",
            note="",
            job=SimpleNamespace(),
            current_study_context=captured,
        )
    assert decisions[0]["population_criterion_acceptances"] == ["c1", "c2"]


def test_acceptance_endpoint_rejects_arbitrary_approval_and_scope_fields(
    acceptance_fixture, monkeypatch
):
    from fastapi.testclient import TestClient
    from easyicu.webserver import agent_pipeline_runs
    from easyicu.webserver.app import app
    from easyicu.webserver.routes import pi_copilot

    service, record, run_id, _, _ = acceptance_fixture
    monkeypatch.setattr(pi_copilot, "get_pi_copilot_service", lambda: service)
    proposal = agent_pipeline_runs.pending_review(run_id)["population_acceptance"]
    body = dict(
        project_id=record.project_id,
        run_id=run_id,
        expected_revision=record.binding.study_revision,
        proposal_sha256=proposal["proposal_sha256"],
        criterion_id="c1",
    )
    client = TestClient(app)
    endpoint = f"/api/copilot/pi/sessions/{record.session_id}/population-acceptance"
    for field in ["cohort", "approval_allowed", "grants", "accepted_criteria"]:
        assert client.post(endpoint, json={**body, field: True}).status_code == 422
    response = client.post(endpoint, json=body)
    assert response.status_code == 200, response.text
    assert response.json()["next_action"] == "review_complete_plan"
