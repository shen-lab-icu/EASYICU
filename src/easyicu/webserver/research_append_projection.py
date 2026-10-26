"""Bounded browser projection of an append beside the existing study plan."""
from __future__ import annotations

from easyicu.research_agent.planning.family_spec.baseline_association_template import LABELS
from easyicu.research_agent.schema import AnalysisPlan
from easyicu.webserver import run_artifact_disclosure, study_contexts
from easyicu.webserver.agent_review_recovery import _write
from easyicu.webserver.pi_copilot.plan_projection import project_plan_conversation_preview


def _step_copy(step):
    methods = {
        "append_exposure_definition": "从已封存测量值推导另一种暴露定义。",
        "append_exposure_absolute_risk": "报告新暴露定义对应的绝对风险。",
        "adjusted_association_figure": "用原有估计值绘制关联图，不重拟合模型。",
        "scientific_reporting": "汇集来源、队列、结果和局限，形成论文分析材料。",
        "restricted_cubic_spline_sensitivity": "增加指定协变量函数形式的敏感性分析。",
        "first_stay_association": "增加仅保留首次入住的敏感性分析。",
        "one_stay_per_patient_association": "增加每位患者仅一次入住的敏感性分析。",
    }
    zh = methods.get(step.method)
    if step.scientific_action_id == "association.effect_modification":
        term = step.model_requirements[0].effect_modifiers[-1]
        zh = f"按 {term.name} 分层，报告各层估计值及同一模型的交互检验。"
    if step.method == "append_exposure_definition":
        zh = f"将 {step.inputs[-1]} 作为连续暴露，采用预设三结点样条；保留原定义的结果。"
    if step.functional_form_spec:
        zh = f"对 {step.functional_form_spec.target_column} 增加样条与线性形式的敏感性比较。"
    if zh is None:
        zh = "另行报告新暴露定义的关联估计，保留此前结果。"
    return dict(step_id=step.step_id, intent=step.intent, intent_zh=zh)


def append_block(child, *, notice=None):
    en, zh = LABELS[child.authority.kind]
    source = AnalysisPlan.model_validate_json(child.authority.source_plan_json)
    source_preview = project_plan_conversation_preview(source.model_dump(mode="json"))
    return dict(kind=child.authority.kind, source_run_id=child.authority.source_run_id,
                source_run_sha256=child.authority.source_run_sha256, label_en=en, label_zh=zh,
                source_notice=notice or child.record["source_notice"],
                source_notice_zh="这次追加沿用原结果的数据版本，原有分析和结果保持不变。",
                source_preview=source_preview,
                steps=[_step_copy(step) for step in child.plan.steps],
                execution_tier=getattr(child.plan.design_selection, "execution_tier", "article"))


def publish_append_projection(child, *, result=None):
    from easyicu.webserver.agent_pipeline_runs import _artifact_record, _table_projection, _figure_projection
    from .research_append_history import merge_result_figures

    from .research_append_source import read_json

    merge_result_figures(child)
    state = child.checkpoint.state
    stop = read_json(child.run_dir / "append_stop.json") if (child.run_dir / "append_stop.json").exists() else {}
    pending = state in {"pending", "approved_pending_execution"}
    complete = state == "completed"
    if complete and result is None:
        from .research_append_source import read_json

        result = read_json(child.run_dir / "append_result.json")
    study = child.record["study"]
    block = append_block(child, notice=(result or {}).get("source_notice"))
    checks = {"execution_complete": complete, "analysis_validated": complete,
              "evidence_complete": complete, "numeric_verified": False, "artifact_valid": complete,
              "manuscript_ready": False, "paper_authorized": False}
    gate = dict(status="blocked" if not complete else "analysis_only", reportable=False, draft_unlocked=False,
                reason="human_plan_review_required" if pending else "research_agent_pipeline_complete_human_interpretation_required"
                if complete else "human_plan_review_rejected" if state == "rejected" else "research_pipeline_execution_failed",
                checks=[{"id": key, "passed": value} for key, value in checks.items()])
    if stop:
        gate.update(reason=stop["failure_code"], detail=stop)
    requests = [dict(review_id=r.review_id, kind=r.kind, summary=r.summary, authority_sha256=r.authority_sha256,
                     reason_code="operator_plan_approval_required", approval_allowed=True) for r in child.checkpoint.requests] if pending else []
    tables = _table_projection(child.run_dir, plan=child.record["plan"])
    own_steps = {step.step_id for step in child.plan.steps}
    tables["tables"] = [table for table in tables["tables"] if table["produced_by_step"] in own_steps
                       and not table["evidence_id"].startswith(("prior_", "source_"))]
    tables["tables"] = [*child.record.get("prior_tables", []), *tables["tables"]]
    tables["table_count"] = len(tables["tables"])
    tables["append"] = {key: block[key] for key in ("kind", "source_run_id", "label_en", "label_zh")}
    tables["append"]["strata"] = [
        {key: row[key] for key in ("subgroup", "stratum", "estimate", "ci_low", "ci_high", "stratum_n", "interaction_p_value")}
        for summary in (result or {}).get("steps", ()) for row in summary.get("stratum_estimates", ())
    ]
    tables["append"]["source_plan"] = AnalysisPlan.model_validate_json(child.authority.source_plan_json).model_dump(mode="json")
    source = child.record["source_data"]
    context = dict(run_id=child.checkpoint.run_id, study_id=study["id"],
                   scientific_configuration_sha256=study_contexts.scientific_configuration_sha256(study),
                   mode="research_agent_pipeline", engine="easyicu.research_agent.pipeline", run_type="full",
                   question=child.plan.research_question, summary={}, local_first={"uploads": 0},
                   source={key: source[key] for key in ("label", "database") if key in source})
    from .research_append_source import read_json

    payloads = {
        "run_context.json": context, "agent_plan.json": child.record["plan"], "quality_gate.json": {"gate": gate, "quality": []},
        "scientific_plan_review.json": read_json(child.run_dir / "scientific_plan_review.json"),
        "result_tables.json": tables,
        "figure_gallery.json": _figure_projection(child.run_dir),
        "source_run_manifest.json": dict(schema_version="easyicu.web-research-pipeline-projection/1",
            engine="easyicu.research_agent.pipeline", run_id=child.checkpoint.run_id,
            status="human_review_pending" if pending else "completed" if complete else state,
            research_input_state="prepared", pending_reviews=requests, plan_approval_allowed=pending,
            plan_revision_source_run_id=child.authority.source_run_id, append=block, readiness=checks, **stop),
    }
    from .scientific_review_metadata import recorded_review_metadata

    payloads["source_run_manifest.json"]["scientific_review"] = recorded_review_metadata(child.run_dir, source)
    if not run_artifact_disclosure.scan_browser_projection(payloads)["passed"]:
        raise ValueError("Append projection failed the browser privacy boundary.")
    for name, payload in payloads.items():
        _write(child.wrapper / name, payload)
    artifacts = [_artifact_record(child.wrapper / name) for name in payloads]
    _write(child.wrapper / "evidence_ledger.json", dict(schema_version="easyicu.web-research-pipeline-ledger/1",
        run_id=child.checkpoint.run_id, run_type="full", engine="easyicu.research_agent.pipeline",
        status=gate["status"], artifacts=artifacts,
        privacy={"patient_rows_in_projection": False, "path_values_returned": False, "projection_scan_passed": True}))
    return dict(run_id=child.checkpoint.run_id, study_id=study["id"], project_dir=str(child.wrapper),
                gate=gate, human_review_pending=pending, pending_reviews=requests, artifacts=artifacts)
