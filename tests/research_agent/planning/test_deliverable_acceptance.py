"""Bilingual acceptance for reviewed deliverables, producers, and Copilot."""

import json
import hashlib
import shutil
import socket
import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent import PipelineServices, ResearchAgentPipeline
from easyicu.research_agent.agents import family_spec_planner
from easyicu.research_agent.agents.progressive_planner import (
    FAMILY_SPEC_STRATEGY,
    ProgressivePlannerAgent,
)
from easyicu.research_agent.agents.reporting import WriterAgent
from easyicu.research_agent.authority.evidence_store import EvidenceStore
from easyicu.research_agent.cohort import materializer
from easyicu.research_agent.cohort.schema import materialize_locked_analysis_cohort, write_locked_cohort_definition
from easyicu.research_agent.contracts.claim_ceiling import first_result_instructions
from easyicu.research_agent.contracts.product_files import (
    HOST_DELIVERY_KINDS, DeliveryRequest, PlanDeliverables, delivery_requests, host_delivery_omissions,
    requested_deliverable_kinds,
)
from easyicu.research_agent.execution.owner_declaration import require_host_rendered_plan
from easyicu.research_agent.execution.runners.selection import resolve_standard_executor
from easyicu.research_agent.intake.materialized_metadata import implementation_bundle_sha256, stage_materialized_cohort_authority
from easyicu.research_agent.literature import (
    CitationRecord, LiteratureBundle, LiteratureScreeningDecision, manuscript_citable_keys,
)
from easyicu.research_agent.methods.table_one import build_grouped_table_one
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.profiles import get_submission_profile
from easyicu.research_agent.orchestration import progressive_planning
from easyicu.research_agent.orchestration.progressive_planning import question_requirement_outcome
from easyicu.research_agent.orchestration.run_entry import analysis_only_after_input_validation
from easyicu.research_agent.orchestration.workflow import HumanReviewPending, build_pipeline_workflow, human_review_requests_for_plan
from easyicu.research_agent import pipeline as pipeline_module
from easyicu.research_agent.planning.requirement_typing.delivery import bind_confirmed_host_delivery, bind_run_deliverables
from easyicu.research_agent.providers.mocks import PatternScriptedMockLLMClient, ScriptedMockLLMClient
from easyicu.research_agent.reporting import completion, manuscript_sections, pdf_render, write_phase
from easyicu.research_agent.schema import AnalysisPlan, CritiqueReport
from easyicu.webserver.agent_runs import read_run_artifact, read_run_artifact_bytes
from easyicu.webserver import agent_pipeline_runs
from easyicu.webserver.agent_pipeline_runs import _write_projection
from easyicu.webserver.pi_copilot.projections import project_run_outcome
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.research_launch_runtime import _submission_profile_ref, reviewed_article_deliverable_defaults
from tests.research_agent.planning.family_spec_fixtures import (
    ALLOWED_CITATIONS,
    DIRECT_COMPARATORS,
    _context,
    _descriptive_context,
    _descriptive_payload,
    _request,
    _spec_payload,
)
from tests.support.node import run_node
from tests.support.pipeline_contracts import disable_article_contract
from tests.support.typed_export import typed_export
from tests.webserver.copilot.pi_copilot_static_fixtures import (
    _load_guided_pi_module_harness as _load_guided_pi_module_harness,
    _read,
)
from tests.webserver.copilot.research_workflow_fixtures import complete_study


PDF = "manuscript_pdf"
PREVIEW = "article_preview"
REFERENCES = "references"
REQUESTS = {
    PDF: ("Please export a downloadable manuscript PDF", "请提供可下载的 PDF"),
    PREVIEW: ("Generate a browser preview", "请提供网页文章预览"),
    REFERENCES: ("Please include a reference list", "请提供参考文献清单"),
}
BASE = (
    "Among adults in the ICU, how is first-day injury stage associated with in-hospital death after "
    "the landmark, and how does ICU length of stay differ by stage?",
    "在成人 ICU 住院中，首日损伤分期与界标后的住院死亡有何关联？各分期的 ICU 住院时长有何差异？",
)
DESKTOP_BASE = (
    "Among adult ICU stays, describe in-hospital mortality by recorded phenotype.",
    "在成人 ICU 住院中，按记录表型描述住院死亡情况。",
)
WITH_WITHOUT = (
    "compare ICU length of stay in patients with and without elevated injury stage",
    "比较伴与不伴损伤分期升高的患者的 ICU 住院时长",
)
DESKTOP_WITHOUT = (
    "report the outcome counts for stays with and without the recorded phenotype",
    "报告具有和不具有记录表型的住院结局计数",
)
PROSE = ("Synthetic aggregate evidence is described for this study.", "本研究描述 ICU 合成汇总证据。")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def deny(*_args, **_kwargs):
        pytest.fail("unexpected network attempt in offline characterization")

    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)


def _plan(questions, tier, *requirements):
    if tier == "first_result":
        context = _descriptive_context().model_copy(update={"research_question": questions})
        cohort_mode = None
        request = _request(context, cohort_mode=cohort_mode).model_copy(update={"execution_tier": tier})
        payload = _descriptive_payload(request, baseline_variables=["age", "sex"])
    else:
        context = _context().model_copy(update={"research_question": questions})
        context.user_preferences.must_have_outputs = (
            "Use the typed association.ordinal_trend action for the secondary ICU length-of-stay analysis."
        )
        cohort_mode = "predicate_filtered"
        request = _request(context, cohort_mode=cohort_mode)
        payload = _spec_payload(request)
    payload["question_requirements"] = list(requirements)
    llm = ScriptedMockLLMClient([json.dumps(payload)] * 3)
    contract = first_result_instructions() if tier == "first_result" else ""
    attempt = ProgressivePlannerAgent(llm).run_attempt(
        context, planner_strategy=FAMILY_SPEC_STRATEGY,
        allowed_literature_citation_keys=ALLOWED_CITATIONS,
        direct_comparator_literature_keys=DIRECT_COMPARATORS,
        comparison_literature_keys=DIRECT_COMPARATORS,
        enforce_article_contract=True, planning_contract_context=contract,
        required_primary_cohort_selection_mode=cohort_mode,
    )
    assert llm.calls
    assert attempt.output.design_selection.execution_tier == tier
    return context, attempt, contract


def _verify_desktop_plan_ownership(run_dir, tmp_path, plan, evidence):
    # A real synthetic typed export provides the materialized-cohort authority
    # for the separate pre-review ownership check.
    source = typed_export(tmp_path / "typed_export")
    paths = materializer.materialize_to_parquet(
        tmp_path / "materialized", data_path=source, database="miiv",
        static_concepts=("age",), feature_concepts=("lact",), outcome_concepts=("death",),
    )
    parent = run_dir / "cohort.parquet"
    assert stage_materialized_cohort_authority(
        paths["parquet"], parent,
        producer_implementation_sha256=implementation_bundle_sha256((Path(materializer.__file__),)),
    ) is not None
    write_locked_cohort_definition(
        run_dir=run_dir, plan=plan, evidence=evidence, prompt_pack_version="test", llm_signature="test",
    )
    result = materialize_locked_analysis_cohort(run_dir=run_dir, plan=plan, universe_path=parent)
    assert result["status"] == "applied"
    require_host_rendered_plan(plan=plan, run_dir=run_dir)


def _requirements(tmp_path, context, attempt, contract):
    evidence = EvidenceStore(tmp_path)
    findings = question_requirement_outcome(
        context=context, plan=attempt.output, facts=attempt.facts,
        run_dir=tmp_path, planning_contract_context=contract,
    )
    requests = human_review_requests_for_plan(
        findings=findings, plan=attempt.output, evidence=evidence, require_plan_review=True,
    )
    review_file = tmp_path / "question_requirements_review.json"
    review = json.loads(review_file.read_text()) if review_file.is_file() else {}
    return evidence, findings, requests, review


def _quotation(quote, kind):
    return {"id": "r1", "kind": "delivery", "quote": quote, "concepts": [],
            "coverage": "plan", "deliverable_kinds": [kind]}


def _literature(question):
    return LiteratureBundle(
        research_question=question,
        citations=[CitationRecord(key="synthetic_ref", title="Synthetic study reference", year="2020")],
        screening_decisions=[LiteratureScreeningDecision(
            citation_key="synthetic_ref", source="synthetic_test",
            disposition="include", evidence_role="related_context",
            rationale="Synthetic case fixture.",
        )],
    )


def _status(tmp_path, context, plan, stop_after_analysis, manuscript_ready):
    gates = {
        "completion_status": "manuscript_ready" if manuscript_ready else "analysis_only",
        "execution_complete": True, "analysis_validated": True, "numeric_verified": manuscript_ready,
        "evidence_complete": manuscript_ready, "manuscript_ready": manuscript_ready,
        "manuscript_requested": not stop_after_analysis,
    }
    completion.apply_host_delivery_completion(gates, plan=plan, run_dir=tmp_path)
    status = completion.run_status_payload(
        gates=gates,
        research_question=context.research_question, code_version="synthetic",
        stop_after_analysis=stop_after_analysis, writer_probe_mode=False,
        writer_probe_failed_steps=[], force_diagnostic_only=False,
    )
    (tmp_path / "run_status.json").write_text(json.dumps(status))
    return status


def _publish(tmp_path, monkeypatch, context, plan, evidence, config, findings, language, renders, *,
             literature_factory=_literature):
    def fake_tex_process(**kwargs):
        tex_path = kwargs["tex_path"]
        bib_path = kwargs["bib_path"]
        assert bool(bib_path) == bool(REFERENCES in plan.deliverables.requested and manuscript_citable_keys(literature))
        pdf_path = tmp_path / "manuscript_scaffold.pdf"
        renders.append("pdf")
        pdf_path.write_bytes(b"%PDF-1.4\n% synthetic renderer response\n%%EOF\n")
        receipt = pdf_render._write_receipt(
            output_dir=tmp_path, tex_path=tex_path, bib_path=bib_path, pdf_path=pdf_path,
            engine="synthetic", draft_watermark=True,
        )
        return pdf_render.PDFRenderResult(
            pdf_path=pdf_path, log_path=None, receipt_path=receipt, engine="synthetic",
            success=True, notes=[],
        )

    monkeypatch.setattr(write_phase, "render_pdf_for_run", fake_tex_process)
    pipeline = SimpleNamespace(
        _config=config, _enable_latex=config.enable_latex, _enable_pdf_render=config.enable_pdf_render,
        _latex_venue_template=config.latex_venue_template, _latex_draft_watermark=True,
        _enable_causal_audit=False, _enable_reporting_checklist=False, _enable_reviewer_round=False,
    )
    stop = analysis_only_after_input_validation(pipeline, False, None)
    report_args = dict(run_dir=tmp_path, run_id=tmp_path.name, manuscript_title="Synthetic study",
                       manuscript_authors=None, emit_progress=lambda *a, **kw: None)
    if stop:
        result = write_phase.run_write_phase(
            pipeline, plan_result=SimpleNamespace(plan=plan), execute_result=SimpleNamespace(plan=plan),
            stop_after_analysis=stop, run_language=language, **report_args,
        )
        assert "results_report.md" in result.bound_path.read_text()
        return _status(tmp_path, context, plan, stop, False)

    # Exercise the production Writer agent with canned replies. The synthetic
    # upstream fixture controls the successful-analysis and evidence-binding inputs.
    prose = PROSE[language == "zh"]
    rules = []
    special = {
        "title": "# Synthetic cohort analysis\n\n**Keywords:** ICU, synthetic cohort, landmark",
        "abstract": ("**Background:** A synthetic cohort was defined.\n\n"
                     "**Methods:** The fixture staged a landmark comparison.\n\n"
                     "**Results:** Aggregate outputs were recorded.\n\n"
                     "**Conclusions:** ICU landmark timing bounds interpretation of the study."),
        "conclusion": ("ICU 解释仅适用于合成界标队列。" if language == "zh"
                       else "Interpretation is bounded to the synthetic landmark cohort and its observed study window."),
    }
    for spec in manuscript_sections.manuscript_section_specs(plan):
        subsections = "".join(f"\n\n### {heading}\n\n{prose}" for heading in spec.required_subsections)
        rules.append((f"Write ONLY the **{spec.section_name}** section", [special.get(spec.key, prose + subsections)] * 4))
    llm = PatternScriptedMockLLMClient(rules, default=prose)
    writer_context = context.model_copy(update={"cohort": context.cohort.model_copy(update={"database": "miiv"})})
    draft = WriterAgent(llm, language=language, nature_writing_enabled=False).run(
        context=writer_context, evidence_ids=[], analysis_plan=plan,
    )
    assert llm.calls
    assert PROSE[language == "zh"] in draft
    literature = literature_factory(context.research_question)
    if "synthetic_ref" in manuscript_citable_keys(literature):
        draft += f"\n\n{prose} [@synthetic_ref]\n"
    bound_path = tmp_path / "manuscript_scaffold_bound.md"
    bound_path.write_text(draft)
    evidence.register_file(kind="log", description="Synthetic bound manuscript", source_path=bound_path,
                           evidence_id="manuscript_scaffold_bound", producer="pipeline", generation_mode="system")
    if literature is not None:
        (tmp_path / "preplan_literature_bundle.json").write_text(literature.model_dump_json())
    table_step = next(step for step in plan.steps if step.table_one_spec)
    spec = table_step.table_one_spec
    labels = list(spec.group_levels) * 20
    columns = {var.name: [i % 2 for i in range(len(labels))] for var in spec.variables}
    frame = pd.DataFrame({**columns, spec.group_by: labels})
    table_path = tmp_path / "table_one.csv"
    build_grouped_table_one(frame, spec).to_csv(table_path, index=False)
    records = [evidence.register_file(
        kind="table", description="Synthetic aggregate Table 1", source_path=table_path,
        evidence_id="table_one", produced_by_step=table_step.step_id,
        producer="pipeline", generation_mode="deterministic_standard",
    )]
    write_phase._persist_reader_artifacts(
        bound=draft, numeric_binding_map={}, run_dir=tmp_path, evidence=evidence, findings=findings,
        context=context, plan=plan, literature=literature, evidence_records=records,
    )
    write_phase._persist_manuscript_critique(
        critique=CritiqueReport(status="pass", reviewer="synthetic"),
        run_dir=tmp_path, evidence=evidence, producer="pipeline",
    )
    write_phase._publish_and_audit_manuscript(
        pipeline, bound=draft, bound_path=bound_path, context=context, plan=plan,
        current_verified_evidence_records=records, evidence=evidence, findings=findings,
        literature=literature, per_step_records=[], repro_envelope=None, **report_args,
    )
    return _status(tmp_path, context, plan, stop, True)


def _render(latest, language, *, session_condition="ready"):
    node = shutil.which("node")
    assert node
    shell = _read("js/screens-guided-pi.js")
    # Run the shell's actual dependency wiring with the synthetic session state.
    stale_guard = shell[shell.index("  function sessionIsStale()"):shell.index("  function iconHtml(")]
    outcome_wiring = shell[shell.index("  const RUN_OUTCOME ="):shell.index("  const ACTIVITY =")]
    js = run_node(node, f"""
      global.window = {{EU_LANG:{language!r}, requestAnimationFrame: callback => callback()}};
      global.document = {{querySelector:()=>null, readyState:'loading', addEventListener() {{}}}};
      global.fetch = () => {{ throw new Error('Unexpected network request'); }};
      eval({_read("js/api.js")!r});
      eval({_read("js/html-escape.js")!r});
      eval({_read("js/screens-guided-pi-resources.js")!r});
      eval({_read("js/screens-guided-pi-run-outcome.js")!r});
      eval({_read("js/screens-guided-pi-events.js")!r});
      const condition = {session_condition!r};
      const handlers = {{}};
      const workflow = {{stages:[{{id:'analysis', status:'complete'}}]}};
      const latest = {json.dumps(latest)};
      const state = {{
        busy: condition === 'busy', childJobId: condition === 'child' ? 'other-job' : '',
        session: condition === 'missing' ? null : {{session_id:'ward-session', stale:{{stale:condition === 'stale'}}}},
        host: {{addEventListener(name, handler) {{ handlers[name] = handler; }}, querySelector:()=>null}},
        latestRun: latest, workflow, draft:'', sessionSelectionRevision:1,
      }};
      const labels = [];
      const projectId = () => 'ward-project';
      const api = () => window.EU_API;
      const esc = window.EU_HTML.esc;
      const resources = window.EU_GUIDED_PI_RESOURCES.create({{esc}});
      const MODULES = window.EasyICU.guidedPi;
      const tr = (en,zh) => {language!r} === 'zh' ? zh : en;
      const iconHtml = () => '';
      const manuscriptNotPassedText = String;
      const resourceButton = (row,label) => {{ labels.push(label); return resources.button(row,label); }};
      const previewWorkflowContext = () => workflow;
      const errorText = String;
      const recordHostAction = () => {{ throw new Error('Unexpected host action'); }};
      const render = () => {{}};
      {stale_guard}
      {outcome_wiring}
      const owner = RUN_OUTCOME;
      const html = owner.render(latest, workflow);
      const button = html.match(/<button[^>]*data-gpi-article-run[^>]*>/);
      if (button) {{
        window.EU_GUIDED_PI_EVENTS.create({{state, projectId, RUN_OUTCOME:owner, render() {{}}}}).wire();
        handlers.click({{target:{{closest: selector => selector === '[data-gpi-article-run]' ? button[0] : null}}}});
      }}
      process.stdout.write(JSON.stringify({{
        html, rows:owner.collection(latest, workflow), shelf:owner.renderShelf(latest, workflow),
        articlePrompt:owner.articleRunPrompt?.(latest), labels,
        articleClick:state.followUpDraft ? {{
          draft:state.draft, intent:state.pendingEntryIntent, origin:state.followUpDraft.origin,
        }} : null,
      }}));
    """, check=False)
    assert js.returncode == 0, js.stderr
    return json.loads(js.stdout)


def _confirmation(run_dir, monkeypatch, language, profile, question, requests):
    study = {**complete_study(), "question": question}
    pending = HumanReviewPending(
        run_id=run_dir.name, thread_id="synthetic-thread", run_dir=str(run_dir), requests=requests,
    )
    entry = SimpleNamespace(
        pending=pending, study=study, credential_source="pi_verified", budget_mode="full_reviewed",
        provider={}, wrapper_dir=run_dir.parent.parent,
    )
    # Supply the upstream scientific review verdict. The Web owner still
    # extracts the exact sealed plan and adjudicates the actual requests.
    scientific_review = {
        "status": "ready_for_approval", "approval_allowed": True, "score": 90,
        "schema_version": agent_pipeline_runs.CURRENT_SCIENTIFIC_REVIEW_SCHEMA_VERSION,
        "findings": [],
    }
    with monkeypatch.context() as patch:
        patch.setattr(agent_pipeline_runs, "_PENDING_REVIEWS", {pending.run_id: entry})
        patch.setattr(agent_pipeline_runs, "_load_pending_scientific_review", lambda *_: scientific_review)
        authority = agent_pipeline_runs.pending_review(pending.run_id)
    assert authority is not None
    assert authority["deliverables"] == requests[0].payload["plan_review_authority"]["plan_payload"]["deliverables"]
    authority["submission_profile_ref"] = profile.ref
    run = {
        "run_id": pending.run_id, "run_type": "full", "engine": "easyicu.research_agent.pipeline",
        "gate_status": "blocked", "run_status": "human_review_pending",
        "pending_review_reason_codes": [row["reason_code"] for row in authority["requests"]],
        "artifact_names": ["agent_plan.json", *authority["question_requirements_recorded"]],
    }
    workflow = build_research_workflow_snapshot(
        study=study, active_export_present=True, active_job=None, latest_run=run, plan_review_authority=authority,
    )
    assert workflow.plan_review_summary["deliverables"] == {
        **authority["deliverables"], "article_required": authority["deliverables_require_article"],
    }
    js = run_node(shutil.which("node"), f"""
      global.window = {{}};
      eval({_read("js/screens-guided-pi-plan-actions.js")!r});
      eval({_read("js/screens-guided-pi-confirmation-data.js")!r});
      eval({_read("js/screens-guided-pi-question-requirements.js")!r});
      eval({_read("js/screens-guided-pi-confirmation.js")!r});
      const owner = window.EU_GUIDED_PI_CONFIRMATION.create({{
        tr:(en,zh)=>{language!r} === 'zh' ? zh : en, esc:String, iconHtml:()=>'', resourceButton:row=>row.artifact, manuscriptNotPassedText:String,
        sessionIsStale:()=>false, busy:()=>false, session:()=>({{}}),
        workflow:()=>({workflow.model_dump_json()}),
      }});
      process.stdout.write(JSON.stringify(owner.workflowConfirmationHtml()));
    """, check=False)
    assert js.returncode == 0, js.stderr
    return json.loads(js.stdout)


def _web_result(run_dir, context):
    wrapper = run_dir.parent.parent
    result = _write_projection(
        wrapper_dir=wrapper, study={"id": "synthetic", "question": context.research_question},
        provider={}, acquisition=None, run_dir=run_dir,
    )
    payloads = {
        item.name: json.loads(item.read_text())
        for item in wrapper.glob("*.json") if item.name != "evidence_ledger.json"
    }
    latest = project_run_outcome({"ok": True, **result, "artifact_payloads": payloads})
    return wrapper, result, payloads, latest


def _run_approved_handoff(context, plan, evidence, findings, write):
    handoff = SimpleNamespace(context=context, plan=plan, evidence=evidence, aborted_result=None)
    workflow = build_pipeline_workflow(
        plan_invoker=lambda: handoff,
        human_review_invoker=lambda state: human_review_requests_for_plan(
            findings=findings, plan=state.plan, evidence=state.evidence, require_plan_review=True,
        ),
        # The delivery matrix supplies an upstream successful execution verdict.
        execute_invoker=lambda state: SimpleNamespace(plan=state.plan),
        write_invoker=lambda state, executed: write(executed.plan),
        finalise_invoker=lambda _state, _executed, written: written,
    )
    pause = workflow.start()
    assert workflow.state == "paused"
    outcome = workflow.resume([
        {"review_id": row.review_id, "authority_sha256": row.authority_sha256,
         "decision": "approved", "reviewer": "Synthetic operator", "decided_at": "2026-01-01T00:00:00Z"}
        for row in pause.requests
    ])
    assert workflow.state == "completed"
    assert {row["decision"] for row in outcome.human_review_decisions} == {"approved"}
    return outcome.final_result


EXCLUSIONS = {
    PDF: ("Do not provide a PDF", "请勿提供 PDF"),
    PREVIEW: ("Please do not generate an online preview", "请不要生成网页文章预览"),
    REFERENCES: ("Do not export the bibliography", "请勿导出参考文献列表"),
}
LABELS = {
    PDF: ("PDF", "PDF"),
    PREVIEW: ("Online preview", "在线预览"),
    REFERENCES: ("References", "参考文献"),
}
ARTIFACTS = {
    PDF: ("manuscript_scaffold.pdf", "manuscript_pdf_receipt.json"),
    PREVIEW: ("manuscript_draft.json", "manuscript_provenance.json", "article_report.json"),
    REFERENCES: ("manuscript_scaffold.bib",),
}

MATRIX = [
    pytest.param(tier, language, kind, wording, id=f"{tier}-{language}-{kind}-{wording}")
    for tier in ("first_result", "article")
    for language in ("en", "zh")
    for kind in (PDF, PREVIEW, REFERENCES)
    for wording in ("explicit", "with_without", "negated", "unmentioned")
]


def _save_plan(run_dir, plan, profile):
    path = run_dir / "analysis_plan.json"
    raw = plan.model_dump_json().encode()
    path.write_bytes(raw)
    (run_dir / "manifest.json").write_text(json.dumps({
        "submission_profile_ref": profile.ref,
        "current_plan_authority": {"relative_path": path.name, "sha256": hashlib.sha256(raw).hexdigest()},
    }))


def _assert_card_choice(card, kind, expected):
    field = re.search(rf'<select name="{kind}"[^>]*>(.*?)</select>', card, re.DOTALL)
    assert field and f'<option value="{expected}" selected>' in field.group(1), (kind, card)


def _fixture_profile(tier):
    return get_submission_profile(_submission_profile_ref(
        budget_mode="full_reviewed", runner_kind="host" if tier == "first_result" else "docker", live_pubmed=False,
    ))


@pytest.mark.parametrize(("tier", "language", "kind", "wording"), MATRIX)
def test_delivery_request(tmp_path, monkeypatch, tier, language, kind, wording):
    zh = language == "zh"
    first = tier == "first_result"
    question = (DESKTOP_BASE if first else BASE)[zh]
    requirements = []
    positive = wording in {"explicit", "with_without"}
    excluded = wording == "negated"
    if positive:
        quote = REQUESTS[kind][zh]
        if wording == "with_without":
            clinical = (DESKTOP_WITHOUT if first else WITH_WITHOUT)[zh]
            mixed = f"{quote}; {clinical}"
            assert not requested_deliverable_kinds(mixed)
            question += f" {mixed}."
        else:
            question += f" {quote}."
        requirements.append(_quotation(quote, kind))
    elif excluded:
        question += f" {EXCLUSIONS[kind][zh]}."
        assert not requested_deliverable_kinds(EXCLUSIONS[kind][zh])

    context, attempt, contract = _plan(question, tier, *requirements)
    profile = _fixture_profile(tier)
    run_dir = tmp_path / "web" / "pipeline" / "run_synthetic"
    run_dir.mkdir(parents=True)
    config = PipelineConfig(
        workdir=run_dir, enable_pdf_render=profile.generate_manuscript is not False,
        default_plan_deliverables=reviewed_article_deliverable_defaults(profile, budget_mode="full_reviewed"),
        **{**profile.as_pipeline_options(), "require_human_plan_review": True},
    )
    plan = bind_run_deliverables(
        attempt.output, question=question, defaults=config.default_plan_deliverables,
        edit=config.bound_plan_deliverables,
    )
    attempt = SimpleNamespace(output=plan, facts=attempt.facts)
    evidence, findings, requests, review = _requirements(run_dir, context, attempt, contract)
    card = _confirmation(run_dir, monkeypatch, language, profile, question, requests)
    approve_label = ("批准并开始分析" if zh else "Approve and start analysis")
    assert ("交付物" if zh else "Deliverables") in card
    if first and positive:
        _assert_card_choice(card, kind, "yes")
        assert "data-gpi-deliverables-article hidden" not in card
        assert ("另行启动论文任务" if zh else "Start a separate article run") in card
        assert approve_label not in card
        assert requests[0].payload["reason"] == "first_result_delivery_requires_article"
        assert requests[0].payload["deliverable_kinds"] == [kind]
        assert requests[0].payload["approval_allowed"] is False
        (judgment,) = review["judged"]
        assert judgment["disposition"] == "not_covered"
        assert judgment["owner_step_ids"] == []
        with pytest.raises(ValueError, match="scientific-stop request cannot be approved"):
            _run_approved_handoff(
                context, plan, evidence, findings,
                lambda _plan: pytest.fail("the tier stop reached delivery without a revised plan"),
            )
        # The existing inline choice can keep the analysis in this tier.
        selection = PlanDeliverables(requested=(), confirmed=True, source="user_edit")
        run_dir = run_dir.with_name("run_analysis_only")
        run_dir.mkdir()
        old_review_digest = requests[0].authority_sha256
        config = config.with_overrides(workdir=run_dir, bound_plan_deliverables=selection.model_dump(mode="json"))
        plan = bind_run_deliverables(
            plan, question=question, defaults=config.default_plan_deliverables,
            edit=config.bound_plan_deliverables,
        )
        attempt = SimpleNamespace(output=plan, facts=attempt.facts)
        evidence, findings, requests, review = _requirements(run_dir, context, attempt, contract)
        assert requests[0].authority_sha256 != old_review_digest
        card = _confirmation(run_dir, monkeypatch, language, profile, question, requests)

    expected = set() if first else set(HOST_DELIVERY_KINDS) - ({kind} if excluded else set())
    assert approve_label in card
    assert {request.payload["reason"] for request in requests} == {"operator_plan_approval_required"}
    assert all(request.payload.get("approval_allowed", True) for request in requests)
    owners = {step.host_delivery_kind: step.step_id for step in plan.steps if step.host_delivery_kind}
    if positive and not first:
        (row,) = review["coverage"]["records"]
        assert row["delivery_kind"] == kind and row["status"] == "covered"
        assert row["owner_step_ids"] == [owners[kind]]
    if first:
        _verify_desktop_plan_ownership(run_dir, tmp_path, plan, evidence)
        scope_notice = ("本次只生成分析表格和图形，不写论文。" if zh else
                        "This run produces analysis tables and figures, not a manuscript.")
        assert scope_notice in card
    scientific_steps = [step for step in plan.steps if step.host_delivery_kind is None]
    assert len(scientific_steps) > 1
    assert any(resolve_standard_executor(step, plan=plan).claimed_by for step in scientific_steps)
    _save_plan(run_dir, plan, profile)
    renders = []
    status = _run_approved_handoff(
        context, plan, evidence, findings,
        lambda approved: _publish(run_dir, monkeypatch, context, approved, evidence, config, findings, language, renders),
    )
    wrapper, result, payloads, latest = _web_result(run_dir, context)
    visible = _render(latest, language)
    assert latest["analysis_outcome"]["status"] == "completed"
    assert status["manuscript_outcome"]["status"] == ("not_requested" if first else "passed")
    ledger = json.loads((wrapper / "evidence_ledger.json").read_text())
    files = {row["name"] for row in ledger["artifacts"]}
    rows = {row["artifact"] for row in visible["rows"]}
    assert "result_tables.json" in rows
    refs = {row["artifact"] for row in latest["artifact_refs"]}
    reader = read_run_artifact(str(wrapper), "manuscript_provenance.json")
    pdf = read_run_artifact_bytes(str(wrapper), "manuscript_scaffold.pdf")
    bib = read_run_artifact_bytes(str(wrapper), "manuscript_scaffold.bib")
    bib_download = re.search(
        r'<a class="gpi-result-download"[^>]+/documents/manuscript_scaffold\.bib\?expected_sha256=[a-f0-9]{64}"'
        r' download="manuscript_scaffold\.bib"',
        visible["shelf"],
    )
    supplied = {
        PDF: pdf["ok"] and pdf["content"].startswith(b"%PDF")
        and "manuscript_scaffold.pdf" in rows and "manuscript_scaffold.pdf" in visible["html"],
        PREVIEW: reader["ok"] and "article_report.json" in rows and "article_report.json" in visible["html"]
        and PROSE[zh] in payloads["manuscript_draft.json"]["markdown_preview"],
        REFERENCES: bib["ok"] and b"synthetic_ref" in bib["content"]
        and "manuscript_scaffold.bib" in rows and bool(bib_download),
    }

    # A checked omission must reach the user even if every other artifact passes.
    if excluded or (first and positive):
        reason = "excluded_by_user" if excluded else "first_result_delivery_requires_article"
        (omission,) = latest.get("deliverable_dispositions", [])
        assert omission == {
            "kind": kind, "reason_code": reason, **({"next_action": "article"} if not excluded else {}),
        }
        message = re.search(rf'<[^>]+data-deliverable-disposition="{kind}"[^>]*>(.*?)</li>',
                            visible["html"], re.DOTALL)
        assert message and LABELS[kind][zh] in message.group(1)
        if excluded:
            assert ("按你的选择省略" if zh else "Excluded at your request") in message.group(1)
        else:
            assert ("本次仅生成首轮分析结果" if zh else "This run produces first results only") in message.group(1)
            assert "data-gpi-article-run" in visible["html"]
            assert LABELS[kind][zh] in visible["articlePrompt"]
            assert visible["articleClick"]["draft"] == visible["articlePrompt"]
    else:
        assert not latest.get("deliverable_dispositions")

    assert set(owners) == expected
    assert {product for product, delivered in supplied.items() if delivered} == expected
    assert renders == (["pdf"] if PDF in expected else [])
    verified = result["gate"].get("host_delivery_steps", [])
    assert all(row["status"] == "verified" for row in verified)
    assert {row["kind"]: row["step_id"] for row in verified} == owners
    for product in HOST_DELIVERY_KINDS:
        choice = "yes" if product in expected else "deferred" if first and positive and product == kind else "no"
        _assert_card_choice(card, product, choice)
        if product not in expected:
            artifacts = set(ARTIFACTS[product])
            assert not artifacts.intersection(files | rows | refs)
            assert all(not (root / name).exists() for root in (run_dir, wrapper) for name in artifacts)
    if excluded and kind == REFERENCES:
        assert "literature_evidence.json" not in files | rows | refs
        if not first:
            assert r"\bibitem{synthetic_ref}" in (run_dir / "manuscript_scaffold.tex").read_text()


@pytest.mark.parametrize("language", ("en", "zh"))
@pytest.mark.parametrize("tier", ("first_result", "article"))
def test_unsplit_pdf_clause_gets_named_stop(tmp_path, monkeypatch, language, tier):
    zh = language == "zh"
    first = tier == "first_result"
    delivery = REQUESTS[PDF][zh]
    clinical = (DESKTOP_WITHOUT if first else WITH_WITHOUT)[zh]
    quote = f"{delivery}; {clinical}"
    requirement = {
        "id": "r1", "kind": "analysis", "quote": quote, "concepts": [],
        "clauses": [delivery, clinical], "coverage": "plan", "deliverable_kinds": [],
    }
    assert requested_deliverable_kinds(delivery) == (PDF,)
    assert not requested_deliverable_kinds(quote)
    base = (DESKTOP_BASE if first else BASE)[zh]
    context, attempt, contract = _plan(f"{base} {quote}.", tier, requirement)
    evidence, findings, requests, review = _requirements(tmp_path, context, attempt, contract)
    reasons = {request.payload["reason"] for request in requests}
    assert "question_requirement_not_covered" in reasons
    assert all(request.payload["approval_allowed"] is False for request in requests)
    with pytest.raises(ValueError, match="scientific-stop request cannot be approved"):
        _run_approved_handoff(
            context, attempt.output, evidence, findings,
            lambda _plan: pytest.fail("a named approval stop reached the Writer"),
        )
    card = _confirmation(tmp_path, monkeypatch, language, _fixture_profile(tier), context.research_question, requests)
    assert delivery in card
    assert ("批准并开始分析" if zh else "Approve and start analysis") not in card


@pytest.mark.parametrize("kind", HOST_DELIVERY_KINDS)
def test_confirmed_card_exclusion_controls_the_producer(tmp_path, monkeypatch, kind):
    context, attempt, contract = _plan(BASE[0], "article")
    selection = PlanDeliverables(
        requested=tuple(product for product in HOST_DELIVERY_KINDS if product != kind),
        confirmed=True, source="user_edit",
    )
    plan = bind_confirmed_host_delivery(attempt.output, selection)
    evidence, findings, _, _ = _requirements(
        tmp_path, context, SimpleNamespace(output=plan, facts=attempt.facts), contract,
    )
    profile = _fixture_profile("article")
    config = PipelineConfig(workdir=tmp_path, enable_pdf_render=True, **profile.as_pipeline_options())
    called = []
    for name in ("build_manuscript_reader", "render_bibtex"):
        original = getattr(write_phase, name)

        def spy(*args, name=name, original=original, **kwargs):
            called.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(write_phase, name, spy)
    renders = []
    _publish(tmp_path, monkeypatch, context, plan, evidence, config, findings, "en", renders)
    assert renders == ([] if kind == PDF else ["pdf"])
    assert ("build_manuscript_reader" in called) is (kind != PREVIEW)
    assert ("render_bibtex" in called) is (kind != REFERENCES)
    for name in ARTIFACTS[kind]:
        assert not (tmp_path / name).exists(), name


def test_first_result_names_all_deliverables_deferred_by_the_tier(tmp_path, monkeypatch):
    question = f"{DESKTOP_BASE[0]} Supply a typeset copy, a browser preview and a bibliography."
    context, attempt, contract = _plan(question, "first_result")
    profile = _fixture_profile("first_result")
    run_dir = tmp_path / "web" / "pipeline" / "run_analyses"
    run_dir.mkdir(parents=True)
    edit = {"requested": [], "confirmed": True, "source": "user_edit"}
    plan = bind_run_deliverables(attempt.output, question=question, defaults=None, edit=edit)
    assert set(plan.deliverables.deferred) == set(HOST_DELIVERY_KINDS)
    evidence, findings, _, _ = _requirements(
        run_dir, context, SimpleNamespace(output=plan, facts=attempt.facts), contract,
    )
    config = PipelineConfig(workdir=run_dir, **profile.as_pipeline_options())
    _save_plan(run_dir, plan, profile)
    _run_approved_handoff(
        context, plan, evidence, findings,
        lambda approved: _publish(run_dir, monkeypatch, context, approved, evidence, config, findings, "en", []),
    )
    _, _, _, latest = _web_result(run_dir, context)
    result = _render(latest, "en")
    assert {row["kind"] for row in latest["deliverable_dispositions"]} == set(HOST_DELIVERY_KINDS)
    assert {row["reason_code"] for row in latest["deliverable_dispositions"]} == {"first_result_delivery_requires_article"}
    assert all(f'data-deliverable-disposition="{kind}"' in result["html"] for kind in HOST_DELIVERY_KINDS)
    assert all(names[0] in result["articlePrompt"] for names in LABELS.values())


REVIEW_QUESTIONS = (
    "Describe how mortality is distributed across the recorded sex categories.",
    "按登记的性别类别描述住院死亡的分布。",
)
REVIEW_CHECK = ("A PDF would be helpful for the care team.", "方便的话请提供 PDF。")


def _pipeline_review(tmp_path, monkeypatch, question, tier, edit):
    disable_article_contract(monkeypatch)
    # This fixture reaches the review boundary without a signed typed export.
    # The matrix verifies desktop executor ownership with that export separately.
    monkeypatch.setattr(pipeline_module, "require_host_plan_before_review", lambda *_args, **_kwargs: None)
    progress = progressive_planning.run_pipeline_progressive_planner

    def plan_without_article_contract(**kwargs):
        kwargs["planner_kwargs"]["enforce_article_contract"] = False
        return progress(**kwargs)

    monkeypatch.setattr(progressive_planning, "run_pipeline_progressive_planner", plan_without_article_contract)
    client = ScriptedMockLLMClient([])
    build_request = family_spec_planner.build_family_spec_request
    labels = {
        "sex": "Recorded sex at admission", "sex=0": "Source category zero",
        "sex=1": "Source category one", "death": "Hospital mortality",
    }

    def script_response(*args, **kwargs):
        request = build_request(*args, **kwargs)
        client.responses.append(json.dumps({
            "schema_version": "easyicu.family_plan_spec/1",
            "family_id": request.family_id,
            "request_sha256": request.request_sha256,
            "adjustment_set": [],
            "baseline_variables": [],
            "distribution_missing_policies": {
                "denominator_policy": "all_declared_rows",
                "missing_exposure_policy": "exclude_from_denominator",
                "missing_outcome_policy": "fail_closed",
            },
            "reader_display_labels": [
                {"key": key, "value": labels[key]}
                for key in (*request.required_reader_label_keys, *request.level_label_keys)
            ],
            "comparator_applications": [],
            "roster_decision_note": "Summarize observed outcomes across recorded groups.",
            "question_requirements": [],
        }))
        return request

    monkeypatch.setattr(family_spec_planner, "build_family_spec_request", script_response)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cohort = tmp_path / "cohort.parquet"
    pd.DataFrame({
        "stay_id": range(1, 13), "sex": [0, 1] * 6, "death": [0, 0, 1, 1] * 3,
    }).to_parquet(cohort, index=False)
    config = PipelineConfig(
        workdir=tmp_path / "pipeline", planner_strategy="family_spec_v1", planner_only=True,
        require_human_plan_review=True, enable_literature=False, enable_memory=False,
        execution_tier=tier, bound_plan_deliverables=edit,
        default_plan_deliverables=reviewed_article_deliverable_defaults(
            _fixture_profile(tier), budget_mode="full_reviewed",
        ),
    )
    pending = ResearchAgentPipeline.from_config(config, services=PipelineServices(llm=client)).run(
        question=question, cohort=cohort, database="miiv", target_outcome="death", primary_exposure="sex",
        user_preferences={"inferred_analysis_family": "descriptive_epidemiology",
                          "covariate_selection": "planner_selectable"},
    )
    assert isinstance(pending, HumanReviewPending)
    assert len(client.calls) == 1
    plan = AnalysisPlan.model_validate_json((Path(pending.run_dir) / "analysis_plan.json").read_text())
    return config, pending, plan


@pytest.mark.parametrize("language", ("en", "zh"))
@pytest.mark.parametrize("wording", ("unmentioned", "check"))
def test_scope_switch_keeps_the_card_dispositions(tmp_path, monkeypatch, language, wording):
    zh = language == "zh"
    question = REVIEW_QUESTIONS[zh]
    if wording == "check":
        question += f" {REVIEW_CHECK[zh]}"
    assert delivery_requests(question)[PDF] is (
        DeliveryRequest.CHECK if wording == "check" else DeliveryRequest.UNMENTIONED
    )
    with monkeypatch.context() as patch:
        _, stopped, _ = _pipeline_review(tmp_path / "selected", patch, question, "first_result", {
            "requested": [PDF], "confirmed": True, "source": "user_edit",
        })
    (delivery_stop,) = [
        row for row in stopped.requests if row.payload["reason"] == "first_result_delivery_requires_article"
    ]
    config, approved_scope, plan = _pipeline_review(
        tmp_path / "analysis-scope" / "web", monkeypatch, question, "first_result",
        {"requested": [], "deferred": [PDF], "excluded": [REFERENCES], "confirmed": True, "source": "user_edit"},
    )
    assert plan.deliverables.deferred == (PDF,)
    assert plan.deliverables.excluded == (REFERENCES,)
    assert not any(step.host_delivery_kind for step in plan.steps)
    assert all(row.authority_sha256 != delivery_stop.authority_sha256 for row in approved_scope.requests)
    assert all(row.payload["reason"] != "first_result_delivery_requires_article" for row in approved_scope.requests)

    run_dir = Path(approved_scope.run_dir)
    _save_plan(run_dir, plan, _fixture_profile("first_result"))
    context = SimpleNamespace(research_question=question)
    status = _publish(run_dir, monkeypatch, context, plan, EvidenceStore(run_dir), config, [], language, [])
    assert status["analysis_outcome"]["status"] == "completed"
    _, _, _, latest = _web_result(run_dir, context)
    assert {row["kind"]: row["reason_code"] for row in latest["deliverable_dispositions"]} == {
        PDF: "first_result_delivery_requires_article", REFERENCES: "excluded_by_user",
    }
    visible = _render(latest, language)
    assert ("扩展成完整论文" if zh else "Extend to a full article") in visible["html"]
    assert "PDF" in visible["articlePrompt"]


@pytest.mark.parametrize("language", ("en", "zh"))
def test_article_pipeline_keeps_card_exclusions(tmp_path, monkeypatch, language):
    question = REVIEW_QUESTIONS[language == "zh"]
    _, _, plan = _pipeline_review(tmp_path, monkeypatch, question, "article", {
        "requested": [PDF], "excluded": [PREVIEW], "confirmed": True, "source": "user_edit",
    })
    assert plan.deliverables.excluded == (PREVIEW,)
    assert [step.host_delivery_kind for step in plan.steps if step.host_delivery_kind] == [PDF]


@pytest.mark.parametrize(("language", "tier", "question", "requested"), [
    ("en", "first_result", "Summarize overnight ward transfers. A browser preview might help at handover.", []),
    ("zh", "article", "汇总夜间转科情况。交班时也许用得上在线预览。", [PDF]),
])
def test_legacy_edit_names_unanswered_delivery_without_attributing_a_no(language, tier, question, requested):
    draft = AnalysisPlan(research_question=question, steps=[])
    if tier == "first_result":
        _, attempt, _ = _plan(question, tier)
        draft = attempt.output.model_copy(update={"deliverables": None})
    assert draft.deliverables is None
    assert delivery_requests(question)[PREVIEW] is DeliveryRequest.CHECK
    plan = bind_run_deliverables(draft, question=question, defaults=HOST_DELIVERY_KINDS, edit={
        "requested": requested, "confirmed": True, "source": "user_edit",
    })
    selection = plan.deliverables
    assert selection.requested == tuple(requested)
    assert selection.deferred == ()
    assert selection.excluded == (PREVIEW,)
    serialized = AnalysisPlan.model_validate_json(plan.model_dump_json()).model_dump(mode="json")
    omissions = host_delivery_omissions(serialized)
    assert omissions == [{"kind": PREVIEW, "reason_code": "delivery_choice_unanswered"}]
    html = _render(_result_with_artifacts(
        "manuscript_provenance.json", report_revision_ready=True, deliverable_dispositions=omissions,
    ), language)["html"]
    message = re.search(r'data-deliverable-disposition="article_preview"[^>]*>(.*?)</li>', html)
    assert message and LABELS[PREVIEW][language == "zh"] in message.group(1)
    assert ("未收到选择，本次未生成" if language == "zh" else "No choice was recorded, so this item was not produced") in message.group(1)
    assert "manuscript_provenance.json" not in html


@pytest.mark.parametrize(("question", "article_exclusion"), [
    ("Summarize early ward discharges.", "No bibliography."),
    ("汇总病房早期出院情况。", "无需参考文献。"),
])
def test_first_result_delivery_selection_binds_to_article(question, article_exclusion):
    _, attempt, _ = _plan(question, "first_result")
    first_selection = PlanDeliverables(
        requested=(), excluded=(PREVIEW,), deferred=(PDF, REFERENCES), confirmed=True, source="user_edit",
    )
    first = bind_run_deliverables(attempt.output, question=question, defaults=None,
                                  edit=first_selection.model_dump(mode="json"))
    assert set(first.deliverables.deferred) == {PDF, REFERENCES}
    assert not any(step.host_delivery_kind for step in first.steps)
    article_question = f"{question} {article_exclusion}"
    assert delivery_requests(article_question)[REFERENCES] is DeliveryRequest.NOT_REQUESTED
    draft = AnalysisPlan(research_question=article_question, steps=[])
    article = bind_run_deliverables(draft, question=article_question, defaults=HOST_DELIVERY_KINDS,
                                    edit=first.deliverables.model_dump(mode="json"))
    assert article.deliverables.requested == (PDF,)
    assert article.deliverables.deferred == ()
    assert set(article.deliverables.excluded) == {PREVIEW, REFERENCES}
    assert [step.host_delivery_kind for step in article.steps] == [PDF]
    assert {row["reason_code"] for row in host_delivery_omissions(article.model_dump(mode="json"))} == {
        "excluded_by_user",
    }


def _screened_out(question):
    return _literature(question).model_copy(update={"screening_decisions": []})


@pytest.mark.parametrize("language", ("en", "zh"))
@pytest.mark.parametrize("literature_factory", [
    pytest.param(lambda _question: None, id="no-search-bundle"),
    pytest.param(_screened_out, id="zero-included"),
])
def test_default_bibliography_without_citable_references(tmp_path, monkeypatch, language, literature_factory):
    zh = language == "zh"
    question = BASE[zh] + (" 同时报告效应估计。" if zh else " Report the estimated effects as well.")
    context, attempt, contract = _plan(question, "article")
    profile = _fixture_profile("article")
    defaults = reviewed_article_deliverable_defaults(profile, budget_mode="full_reviewed")
    assert REFERENCES in defaults
    run_dir = tmp_path / "web" / "pipeline" / "run_no_citable_keys"
    run_dir.mkdir(parents=True)
    plan = bind_run_deliverables(attempt.output, question=question, defaults=defaults, edit=None)
    evidence, findings, _, _ = _requirements(
        run_dir, context, SimpleNamespace(output=plan, facts=attempt.facts), contract,
    )
    config = PipelineConfig(workdir=run_dir, enable_pdf_render=True, **profile.as_pipeline_options())
    _save_plan(run_dir, plan, profile)
    status = _publish(
        run_dir, monkeypatch, context, plan, evidence, config, findings, language, [],
        literature_factory=literature_factory,
    )
    assert status["status"] == "manuscript_ready", status["gates"].get("host_delivery_failure")
    wrapper, result, _, latest = _web_result(run_dir, context)
    assert {row["kind"]: row["status"] for row in result["gate"]["host_delivery_steps"]} == {
        kind: "verified" for kind in HOST_DELIVERY_KINDS
    }
    assert latest["deliverable_dispositions"] == [{"kind": REFERENCES, "reason_code": "no_citable_references"}]
    bib = read_run_artifact_bytes(str(wrapper), "manuscript_scaffold.bib")
    assert bib["ok"] and bib["content"] == b"@comment{No citable references}\n"
    visible = _render(latest, language)
    assert ("本次没有可引用的参考文献" if zh else "This run has no citable references") in visible["html"]
    assert ("空的 BibTeX 文件" if zh else "Empty BibTeX file") in visible["html"]
    assert 'download="manuscript_scaffold.bib"' in visible["shelf"]


def _result_with_artifacts(*artifacts, **state):
    return {
        "present": True, "run_id": "ward-recovery", "analysis_results_available": True,
        "analysis_validated": True, "research_question": "Summarize outcomes after a ward transfer.",
        "artifact_refs": [
            {"kind": "research_document" if name.endswith((".pdf", ".bib")) else "research_artifact",
             "artifact": name, "run_id": "ward-recovery", "sha256": "c" * 64}
            for name in ("evidence_ledger.json", *artifacts)
        ],
        **state,
    }


@pytest.mark.parametrize("language", ("en", "zh"))
def test_excluded_preview_stays_hidden_after_report_revision(language):
    preview_files = ("article_report.json", "manuscript_draft.json", "manuscript_provenance.json")
    latest = _result_with_artifacts(
        *preview_files, "manuscript_revision.pdf",
        report_revision_ready=True, report_revision_pdf_ready=True,
    )
    allowed = _render(latest, language)
    visible = _render({
        **latest, "deliverable_dispositions": [{"kind": PREVIEW, "reason_code": "excluded_by_user"}],
    }, language)
    rows = {row["artifact"] for row in visible["rows"]}
    assert 'data-gpi-resource-artifact="manuscript_provenance.json"' in allowed["html"]
    for artifact in preview_files:
        assert artifact in allowed["shelf"]
        assert artifact not in rows
        assert artifact not in visible["html"]
        assert artifact not in visible["shelf"]
        assert artifact not in visible["labels"]
    for labels in [
        ("View article report with figures", "查看含图文章报告"),
        ("View report draft and revision status", "查看报告草稿与修订状态"),
        ("Preview evidence-bound article", "预览证据绑定文章"),
    ]:
        label = labels[language == "zh"]
        assert label in allowed["labels"]
        assert label not in visible["labels"]
    assert "manuscript_revision.pdf" in rows
    assert "manuscript_revision.pdf" in visible["html"]
    assert ("按你的选择省略" if language == "zh" else "Excluded at your request") in visible["html"]


def _deferred_result():
    return _result_with_artifacts(deliverable_dispositions=[
        {"kind": PDF, "reason_code": "first_result_delivery_requires_article", "next_action": "article"},
    ])


@pytest.mark.parametrize("language", ("en", "zh"))
@pytest.mark.parametrize("condition", ("busy", "child", "missing", "stale"))
def test_extend_article_is_disabled_when_unavailable(language, condition):
    visible = _render(_deferred_result(), language, session_condition=condition)
    assert re.search(r"<button[^>]+data-gpi-article-run[^>]+\bdisabled\b", visible["html"])
    assert visible["articleClick"] is None


@pytest.mark.parametrize("language", ("en", "zh"))
def test_acceptance_reaches_download_and_article_click(language):
    latest = _result_with_artifacts("manuscript_scaffold.bib", manuscript_ready=True)
    visible = _render(latest, language)
    assert re.search(
        r'<a class="gpi-result-download" href="/api/copilot/pi/projects/ward-project/runs/ward-recovery/'
        r'documents/manuscript_scaffold\.bib\?expected_sha256=c{64}" download="manuscript_scaffold\.bib"',
        visible.get("shelf", ""),
    )
    visible = _render(_deferred_result(), language)
    button = re.search(r"<button[^>]+data-gpi-article-run[^>]*>", visible["html"])
    assert button and " disabled" not in button.group()
    zh = language == "zh"
    assert visible["articleClick"] == {
        "draft": ("请为这项研究另行启动论文任务，并提供PDF。" if zh else
                  "Start a separate article run for this study with PDF."),
        "intent": "",
        "origin": "host_prompt",
    }
