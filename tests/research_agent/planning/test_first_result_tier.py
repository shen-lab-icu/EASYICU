"""A run may stop at typed evidence without inheriting article obligations."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep, CohortDescriptor, ResearchContext


def context():
    return ResearchContext(
        research_question="Association of a synthetic baseline attribute with a binary outcome.",
        cohort=CohortDescriptor(cohort_name="synthetic", database="synthetic", n_stays=12),
        variables=[],
    )


def plan(tier="first_result", steps=()):
    from easyicu.research_agent.planning.design_selection import ResearchDesignSelection

    candidates = [dict(
        design_id=name, analysis_type="association_study", estimand="Synthetic adjusted association",
        time_zero="Study baseline", observation_window="Fixed synthetic observation window",
        primary_method=method, required_variables=["synthetic_marker", "synthetic_outcome"],
        assumptions=["Recorded baseline precedes outcome"], novelty_positioning="Pre-result design comparison",
        figure_role="Typed result brief", supports="Association estimates", cannot_prove="Causal effects",
        disposition=disposition, decision_reason="Selected by the stated question and timing",
    ) for name, method, disposition in [("main", "logistic regression", "selected"), ("alternative", "stratified regression", "rejected")]]
    return AnalysisPlan(
        research_question=context().research_question, analysis_type="association_study",
        design_selection=ResearchDesignSelection(execution_tier=tier, candidates=candidates), steps=list(steps),
    )


def test_tier_belongs_to_the_plan_and_is_immutable():
    first = plan()
    article = plan("article")
    assert first.design_selection.execution_tier == "first_result"
    assert article.design_selection.execution_tier == "article"
    assert "execution_tier" not in context().model_dump()
    assert AnalysisPlan.model_validate(first.model_dump()).design_selection.execution_tier == "first_result"
    with pytest.raises(ValidationError):
        first.design_selection.execution_tier = "article"


def test_tier_changes_the_approved_plan_digest():
    from easyicu.research_agent.canonical_json import canonical_sha256

    assert canonical_sha256(plan().model_dump()) != canonical_sha256(plan("article").model_dump())
    assert "execution_tier" not in plan("article").model_dump()["design_selection"]  # Legacy approved digests stay valid.


def test_unknown_execution_tier_is_rejected(tmp_path):
    from easyicu.research_agent.orchestration.config import PipelineConfig

    with pytest.raises(ValueError, match="execution_tier"):
        PipelineConfig(workdir=tmp_path, execution_tier="paper")


def test_new_run_can_choose_article_without_changing_the_first_run(tmp_path):
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.run_entry import seal_plan_tier

    first = plan()
    article = seal_plan_tier(first, PipelineConfig(workdir=tmp_path, execution_tier="article"), resumed=False)
    assert first.design_selection.execution_tier == "first_result" and article.design_selection.execution_tier == "article"
    new_first = seal_plan_tier(article, PipelineConfig(workdir=tmp_path, execution_tier="first_result"), resumed=False)
    assert new_first.design_selection.execution_tier == "first_result"


def test_first_result_requires_a_reviewed_design_contract(tmp_path):
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.run_entry import seal_plan_tier

    legacy = AnalysisPlan(research_question="Synthetic association question", steps=[])
    with pytest.raises(ValueError, match="reviewed design selection"):
        seal_plan_tier(legacy, PipelineConfig(workdir=tmp_path, execution_tier="first_result"), resumed=False)


def test_first_result_guardrails_keep_science_but_omit_article_defaults():
    from easyicu.research_agent.planning.scientific_review import render_plan_scientific_guardrails

    text = render_plan_scientific_guardrails(context(), execution_tier="first_result")
    assert "measurement_audit" in text
    assert "host-owned" in text
    assert "two distinct" not in text
    assert "include an executable, source-bound functional-form check" not in text
    assert "two distinct" in render_plan_scientific_guardrails(context())


def test_first_result_playbook_omits_article_displays():
    from easyicu.research_agent.planning.study_design import build_study_design_brief

    brief = build_study_design_brief(context(), execution_tier="first_result")
    assert not brief.main_text_displays
    assert not brief.sensitivity_requirements
    assert any("measurement" in item for item in brief.required_methods)


def test_first_result_readiness_has_no_optional_article_roles(monkeypatch):
    from easyicu.research_agent.planning import publication_readiness
    from easyicu.research_agent.planning.publication_readiness import build_publication_readiness_facts
    from easyicu.research_agent.planning.figure_strategy import build_article_figure_strategy
    from easyicu.research_agent.planning.study_design import build_study_design_brief

    # A cached article playbook cannot reintroduce obligations to this run.
    monkeypatch.setattr(publication_readiness, "build_study_design_brief", lambda study, **kw: build_study_design_brief(study))

    facts = build_publication_readiness_facts(
        context=context(), plan=plan(), figure_strategy=build_article_figure_strategy(context()), sensitivity={},
    )
    assert facts["robustness"]["required_axis_count"] == 0
    assert facts["robustness"]["effect_style_grid_required"] is False
    assert not facts["figure_roles"]["required_roles"]
    assert not facts["content_roles"]["required_roles"]


def test_first_result_cannot_enter_writer_even_with_article_configuration(tmp_path):
    from easyicu.research_agent.reporting.write_phase import run_write_phase

    result = run_write_phase(
        SimpleNamespace(_config=SimpleNamespace(generate_manuscript=True)),
        plan_result=SimpleNamespace(plan=plan()), execute_result=SimpleNamespace(plan=plan()),
        run_dir=tmp_path, run_id="synthetic", stop_after_analysis=False,
        manuscript_title=None, manuscript_authors=None, run_language="en",
        emit_progress=lambda *a, **kw: pytest.fail("writer entered"), force_writer_probe=True,
    )
    assert "results_report.md" in result.bound_path.read_text()
    assert not list(tmp_path.glob("*.tex"))


def test_first_result_refuses_article_only_steps_before_approval(monkeypatch):
    from easyicu.research_agent.execution.owner_declaration import require_host_plan_before_review
    from easyicu.research_agent.execution import owner_declaration

    step = AnalysisStep(step_id="paper", intent="Draft a manuscript", method="report",
                        expected_outputs=["report:manuscript", "table:audit"], measurement_audit_spec={
                            "products": [{"product_id": "audit", "audit": "measurement_missingness"}],
                        })
    proposed = plan(steps=[step])
    pipeline = SimpleNamespace(_runner_kind="docker", _scientific_runtime_projection_sha256=None,
                               _scientific_runtime_authorities=SimpleNamespace(trajectory=None, current_case=None))
    monkeypatch.setattr(owner_declaration, "resolve_standard_executor", lambda *a, **kw: SimpleNamespace(claimed_by="synthetic"))
    with pytest.raises(ValueError, match="first_result"):
        require_host_plan_before_review(pipeline, proposed)


def test_first_result_refuses_an_unowned_step_even_with_docker(monkeypatch):
    from easyicu.research_agent.execution import owner_declaration

    audit = AnalysisStep(step_id="audit", intent="Audit measurements", method="measurement_audit",
                         expected_outputs=["table:audit"], measurement_audit_spec={
                             "products": [{"product_id": "audit", "audit": "measurement_missingness"}],
                         })
    generated = AnalysisStep(step_id="generated", intent="Fit an unsupported model", method="custom_model",
                            expected_outputs=["table:estimates"])
    proposed = plan(steps=[audit, generated])
    pipeline = SimpleNamespace(_runner_kind="docker", _scientific_runtime_projection_sha256=None,
                               _scientific_runtime_authorities=SimpleNamespace(trajectory=None, current_case=None))
    monkeypatch.setattr(owner_declaration, "resolve_standard_executor", lambda step, **kw: SimpleNamespace(
        claimed_by="synthetic_audit" if step.step_id == "audit" else None))
    with pytest.raises(owner_declaration.HostPlanStepNeedsSandbox) as error:
        owner_declaration.require_host_plan_before_review(pipeline, proposed)
    assert error.value.easyicu_safe_diagnostic["step_id"] == "generated"


def test_first_result_can_never_be_manuscript_ready(tmp_path):
    from easyicu.research_agent.authority.evidence_store import EvidenceStore
    from easyicu.research_agent.reporting.readiness import _compute_readiness_gates

    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text("# Synthetic paper\n\n## Results\nA synthetic estimate was obtained from this [registered evidence](evidence/result.csv).\n")
    gates = _compute_readiness_gates(
        context=context(), plan=plan(), per_step_records=[], findings=[],
        evidence=EvidenceStore(tmp_path), run_dir=tmp_path, manuscript_path=manuscript,
        stop_after_analysis=False, execution_paper_eligible=True, plan_authority_verified=True,
    )
    assert gates["manuscript_generated"] is False
    assert gates["manuscript_ready"] is False
    assert gates["execution_paper_eligible"] is False
    assert gates["paper_authorized"] is False


def test_first_result_run_status_never_requests_a_manuscript(monkeypatch, tmp_path):
    from easyicu.research_agent.reporting import readiness
    from easyicu.research_agent.authority.evidence_store import EvidenceStore

    class StatusObserved(Exception):
        pass

    captured = {}

    def observe_status(**kwargs):
        captured.update(kwargs)
        raise StatusObserved

    monkeypatch.setattr(readiness, "build_run_status_payload", observe_status)
    with pytest.raises(StatusObserved):
        readiness.write_readiness_artifacts(
            context=context(), plan=plan(), per_step_records=[], findings=[],
            evidence=EvidenceStore(tmp_path), run_dir=tmp_path,
            manuscript_path=tmp_path / "manuscript.md", stop_after_analysis=False,
        )
    assert captured["stop_after_analysis"] is True


def test_approved_tier_cannot_change_on_resume(tmp_path):
    from easyicu.research_agent.orchestration.config import PipelineConfig
    from easyicu.research_agent.orchestration.run_entry import seal_plan_tier

    first = plan()
    assert seal_plan_tier(first, PipelineConfig(workdir=tmp_path, execution_tier="first_result"), resumed=True) is first
    with pytest.raises(ValueError, match="cannot change"):
        seal_plan_tier(first, PipelineConfig(workdir=tmp_path), resumed=True)


def test_first_result_cannot_enter_manuscript_revision(tmp_path):
    from easyicu.research_agent.reporting.writer_only_migration import prepare_writer_only_migration

    (tmp_path / "analysis_plan.json").write_text(plan().model_dump_json())
    with pytest.raises(ValueError, match="first_result"):
        prepare_writer_only_migration(tmp_path)


def test_landmark_subset_keeps_stated_sensitivities_and_omits_automatic_rcs():
    from easyicu.research_agent.planning.family_spec import (
        FamilyPlanSpec, build_landmark_categorical_skeleton,
    )
    from tests.research_agent.planning.family_spec_fixtures import _context, _request, _spec_payload

    request = _request(_context()).model_copy(update={"execution_tier": "first_result", "functional_form_spec_ids": {}})
    draft = build_landmark_categorical_skeleton(request, FamilyPlanSpec.model_validate(_spec_payload(request)))
    modules = {item.step_id for item in draft.outline.steps}
    assert {"measurement_audit", "adjusted_association", "robustness_replay", "ordinal_trend"} <= modules
    assert not {"report", "visualization"} & modules
    assert not any(name.endswith("_functional_form") for name in modules)


def test_a_planner_cannot_choose_the_run_tier():
    from easyicu.research_agent.agents.plan_payload import (
        PlannerHostOwnedFieldError, _planner_transport_schema, decode_planner_transport_payload,
    )

    assert "execution_tier" not in _planner_transport_schema()["properties"]
    from easyicu.research_agent.planning.design_selection import ResearchDesignSelection
    assert "execution_tier" not in ResearchDesignSelection.model_json_schema()["properties"]
    with pytest.raises(PlannerHostOwnedFieldError):
        decode_planner_transport_payload(plan().model_dump())
    restored = decode_planner_transport_payload({"steps": []}, host_owned_plan=plan().model_dump())
    assert restored["design_selection"]["execution_tier"] == "first_result"
    with pytest.raises(PlannerHostOwnedFieldError):
        decode_planner_transport_payload({"execution_tier": "article"}, host_owned_plan=plan().model_dump())
    changed = plan().model_dump()
    changed["design_selection"]["execution_tier"] = "article"
    with pytest.raises(PlannerHostOwnedFieldError):
        decode_planner_transport_payload(changed, host_owned_plan=plan().model_dump())


def test_first_result_keeps_explicit_sensitivity_review():
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review
    from easyicu.research_agent.planning.sensitivity_authority import PrespecifiedSensitivitySpec
    from easyicu.research_agent.schema import UserPreferences

    stated = context().model_copy(update={"user_preferences": UserPreferences(sensitivity_specs=[
        PrespecifiedSensitivitySpec(spec_id="requested_cases", axis="missing_data", strategy="complete_case", execution_variables=("synthetic_marker",)),
    ])})
    review = build_plan_scientific_review(context=stated, plan=plan())
    assert "REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY" in {item.code for item in review.findings}
    assert not review.approval_allowed


def test_a_replanner_cannot_clear_the_approved_tier_with_null():
    from easyicu.research_agent.agents.plan_payload import decode_planner_transport_payload

    restored = decode_planner_transport_payload({"design_selection": None}, host_owned_plan=plan().model_dump())
    assert restored["design_selection"]["execution_tier"] == "first_result"


def test_a_first_result_review_with_no_findings_is_still_analysis_only(monkeypatch):
    from easyicu.research_agent.planning import publication_readiness
    from easyicu.research_agent.planning.scientific_review import build_plan_scientific_review

    # Even a perfect scientific review cannot grant article authority to this run.
    monkeypatch.setattr(publication_readiness, "tier_review_findings", lambda *_: [])
    review = build_plan_scientific_review(context=context(), plan=plan())
    assert review.approval_allowed and not review.findings
    assert review.status == "analysis_only"
    assert not review.top_journal_candidate


def test_first_result_review_defers_article_positioning_only():
    from easyicu.research_agent.planning.publication_readiness import tier_review_findings

    findings = [SimpleNamespace(code=code) for code in (
        "TOP_JOURNAL_LITERATURE_SEARCH_NOT_ESTABLISHED", "NOVELTY_NOT_ESTABLISHED",
        "NOVELTY_POSITIONING_REVIEW_REQUIRED", "REQUIRED_SENSITIVITY_IS_PROTOCOL_ONLY",
    )]
    assert tier_review_findings(plan(), findings) == findings[-1:]
    assert tier_review_findings(plan("article"), findings) == findings


def test_first_result_has_no_automatic_publication_figures():
    from easyicu.research_agent.planning.figure_plan_mutation import ensure_publication_figure_step_in_plan

    first = plan(steps=[AnalysisStep(step_id="primary", intent="Estimate the association", method="adjusted_association_models",
                                    expected_outputs=["table:primary"])])
    revised, _ = ensure_publication_figure_step_in_plan(plan=first, context=context(), force=True)
    assert revised is first


def test_first_result_requires_a_typed_measurement_audit():
    from easyicu.research_agent.execution.owner_declaration import require_first_result_steps

    with pytest.raises(ValueError, match="measurement_audit"):
        require_first_result_steps(plan())


def test_landmark_first_result_keeps_requested_rcs():
    from easyicu.research_agent.planning.family_spec import FamilyPlanSpec, build_landmark_categorical_skeleton
    from tests.research_agent.planning.family_spec_fixtures import _context, _request, _spec_payload

    request = _request(_context()).model_copy(update={
        "execution_tier": "first_result", "functional_form_spec_ids": {"age": "requested_age_rcs"},
    })
    draft = build_landmark_categorical_skeleton(request, FamilyPlanSpec.model_validate(_spec_payload(request)))
    assert "age_functional_form" in {step.step_id for step in draft.outline.steps}


def test_landmark_first_result_omits_unrequested_robustness():
    from easyicu.research_agent.planning.family_spec import FamilyPlanSpec, build_landmark_categorical_skeleton
    from tests.research_agent.planning.family_spec_fixtures import _context, _request, _spec_payload

    request = _request(_context()).model_copy(update={
        "execution_tier": "first_result", "functional_form_spec_ids": {},
        "complete_case_spec_id": None, "alternate_exposures": [], "first_stay": None,
    })
    draft = build_landmark_categorical_skeleton(request, FamilyPlanSpec.model_validate(_spec_payload(request)))
    assert "robustness_replay" not in {step.step_id for step in draft.outline.steps}


def test_first_result_cannot_publish_a_prepared_manuscript(tmp_path):
    from easyicu.research_agent.reporting.writer_only_migration import publish_writer_only_result, repair_writer_only

    prepared = SimpleNamespace(plan=plan())
    with pytest.raises(ValueError, match="first_result"):
        repair_writer_only(prepared, writer=object())
    with pytest.raises(ValueError, match="first_result"):
        publish_writer_only_result(prepared, None, output_dir=tmp_path / "paper", provider="synthetic",
                                  model="synthetic", provider_summary={}, provider_ledger="")
    assert not (tmp_path / "paper").exists()


def test_first_result_landmark_compiles_with_stated_requirements():
    import json
    from easyicu.research_agent.agents.progressive_planner import FAMILY_SPEC_STRATEGY, ProgressivePlannerAgent
    from easyicu.research_agent.contracts.claim_ceiling import first_result_instructions
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from tests.research_agent.planning.family_spec_fixtures import (
        ALLOWED_CITATIONS, DIRECT_COMPARATORS, _context, _request, _spec_payload,
    )

    study = _context()
    study.user_preferences.must_have_outputs = "Use the typed ordinal trend for the secondary ICU length-of-stay analysis."
    study.user_preferences.sensitivity_specs = [s for s in study.user_preferences.sensitivity_specs if s.axis != "functional_form"]
    request = _request(study).model_copy(update={"execution_tier": "first_result"})
    attempt = ProgressivePlannerAgent(ScriptedMockLLMClient([json.dumps(_spec_payload(request))])).run_attempt(
        study, planner_strategy=FAMILY_SPEC_STRATEGY, allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS, comparison_literature_keys=DIRECT_COMPARATORS,
        enforce_article_contract=True, planning_contract_context=first_result_instructions(),
        required_primary_cohort_selection_mode="predicate_filtered",
    )
    assert attempt.output.design_selection.execution_tier == "first_result"
    assert {"measurement_audit", "ordinal_trend", "robustness_replay"} <= {step.step_id for step in attempt.output.steps}
    assert not any(step.step_id.endswith("_functional_form") for step in attempt.output.steps)
