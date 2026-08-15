"""Science workbench summaries for Agent/Idea provenance surfaces.

This module is a presentation adapter over existing local Agent review
artifacts. It does not change the evidence gate and does not make any run
reportable; it only shapes the existing bounded artifacts into UI-ready
objects inspired by scientific workbench artifact/review panes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from easyicu.research_agent.publication_skills import (
    publication_skill_workbench_cards,
)
from easyicu.webserver import agent_runs, capabilities
from easyicu.webserver.ideas import mining as idea_mining_web


SCHEMA_VERSION = "easyicu.science_workbench/1"

VISUAL_REFERENCE = {
    "source": "public_web_reference",
    "title": "Claude Science artifact/workbench screenshot",
    "article_url": "https://www.testingcatalog.com/early-look-at-anthropics-claude-science-app-for-researchers/",
    "image_url": "https://storage.ghost.io/c/2a/1b/2a1b1782-8506-4d7d-bf53-ad3fb52e2a0f/content/images/size/w2000/2026/07/1c78d0a671cbf1715b3f09a790e6d1a90466de1a-2048x1257.webp",
    "design_cues": [
        "left project/files rail",
        "center artifact preview",
        "right Code / Execution Log / Messages / Environment / Review tabs",
        "visible code provenance next to generated figures",
        "review state kept adjacent to artifact history",
    ],
}

PROTOCOL_SKILLS = [
    {
        "id": "icu_idea_to_picot",
        "title": "Literature idea to ICU estimand",
        "stage": "Idea Mining",
        "route": "ideas",
        "scope": "Turn an external review/editorial trigger into a locked ICU question.",
        "inputs": ["source metadata", "traceable quote", "population", "exposure", "outcome"],
        "outputs": ["locked PICOT question", "allowed claim boundary", "handoff seed"],
        "evidence": ["source citation", "quote hash", "human confirmation"],
    },
    {
        "id": "outcome_blind_feasibility",
        "title": "Outcome-blind feasibility gate",
        "stage": "Feasibility",
        "route": "ideas",
        "scope": "Check denominator, concept availability, windows, and portability before effect estimates.",
        "inputs": ["candidate concepts", "database scope", "time zero", "analysis window"],
        "outputs": ["feasibility ledger", "hold/go decision", "blocker list"],
        "evidence": ["coverage table", "denominator summary", "missingness structure"],
    },
    {
        "id": "agent_artifact_history",
        "title": "Agent artifact history package",
        "stage": "Agent run",
        "route": "agent",
        "scope": "Bundle code, environment, inputs, artifact hashes, and message history proxies.",
        "inputs": ["active export", "run context", "analysis code path", "provider boundary"],
        "outputs": ["artifact_history.json-equivalent", "evidence ledger", "downloadable bundle"],
        "evidence": ["SHA-256", "run_context.json", "evidence_ledger.json"],
    },
    {
        "id": "icu_reviewer_gate",
        "title": "ICU reviewer gate",
        "stage": "Review",
        "route": "agent",
        "scope": "Check citation, numeric, figure, denominator, privacy, and conclusion-safety signals.",
        "inputs": ["quality gate", "manuscript draft", "evidence ledger", "figure gallery"],
        "outputs": ["reviewer gate card", "sign-off checklist", "blocked-claim reasons"],
        "evidence": ["quality_gate.json", "numeric audit", "strict evidence audit"],
    },
    *publication_skill_workbench_cards(),
]

NATIVE_RENDERERS = [
    {
        "id": "concept_coverage_matrix",
        "title": "Concept coverage matrix",
        "artifact_names": ["quality_gate.json", "missingness_audit.json"],
        "route": "agent",
        "description": "Render module/concept coverage and structural absence without patient rows.",
        "privacy": "aggregate_only",
    },
    {
        "id": "cohort_attrition",
        "title": "Cohort attrition",
        "artifact_names": ["cohort_summary.json", "table1_summary.json"],
        "route": "agent",
        "description": "Show denominator, outcome availability, and cohort grouping before claims.",
        "privacy": "aggregate_only",
    },
    {
        "id": "icu_time_lane",
        "title": "ICU time lane",
        "artifact_names": ["run_context.json", "workflow_graph.json"],
        "route": "patient",
        "description": "Use ICU-native timelines for patient/source review; fall back to run phases in Agent review.",
        "privacy": "bounded_preview",
    },
    {
        "id": "claim_evidence_graph",
        "title": "Claim evidence graph",
        "artifact_names": ["manuscript_draft.json", "evidence_ledger.json", "quality_gate.json"],
        "route": "agent",
        "description": "Draw claims, evidence IDs, hashes, and gate decisions as a reviewable graph.",
        "privacy": "no_patient_rows",
    },
]


def build_science_workbench(project_dir: Optional[str] = None) -> Dict[str, Any]:
    """Return UI-ready science-workbench state for a local Agent run.

    ``project_dir`` is optional so the UI can show the design affordance before
    a user opens a real local run. When provided, this function delegates all
    artifact reading to ``agent_runs.read_run_review``.
    """
    review: Optional[Dict[str, Any]] = None
    if project_dir:
        review = agent_runs.read_run_review(project_dir)
        if not review.get("ok"):
            return review

    payloads: Dict[str, Dict[str, Any]] = (
        review.get("artifact_payloads", {}) if review else {}
    )
    artifacts: List[Dict[str, Any]] = review.get("artifacts", []) if review else []
    gate = review.get("gate", {}) if review else {}
    readiness = review.get("readiness", {}) if review else {}
    ledger = payloads.get("evidence_ledger.json", {})
    capability_policy = capabilities.capability_status()
    capability_flags = capability_policy.get("settings") or {}
    reusable_protocols = []
    if bool(capability_flags.get("science_skills_enabled", True)):
        reusable_protocols = [
            row
            for row in PROTOCOL_SKILLS
            if not row.get("setting_key")
            or bool(
                capability_flags.get(
                    str(row["setting_key"]),
                    row.get("default_enabled", True),
                )
            )
        ]

    artifact_history = _artifact_history(
        artifacts=artifacts,
        payloads=payloads,
        ledger=ledger,
        project_dir=review.get("project_dir") if review else None,
    )
    reviewer_gate = _reviewer_gate(
        review=review,
        payloads=payloads,
        gate=gate,
        readiness=readiness,
        ledger=ledger,
    )
    renderers = _native_renderers(payloads, artifacts, gate)
    workflow_scope = _workflow_scope(review, payloads)
    run_summary = _run_summary(
        review=review,
        payloads=payloads,
        artifact_history=artifact_history,
        reviewer_gate=reviewer_gate,
        renderers=renderers,
        workflow_scope=workflow_scope,
    )
    fig5_checklist = _fig5_checklist(
        payloads=payloads,
        artifact_history=artifact_history,
        reviewer_gate=reviewer_gate,
        renderers=renderers,
        workflow_scope=workflow_scope,
    )
    discovery_pipeline = _discovery_pipeline()
    feature_alignment = _feature_alignment(
        artifact_history=artifact_history,
        reviewer_gate=reviewer_gate,
        renderers=renderers,
        workflow_scope=workflow_scope,
        discovery_pipeline=discovery_pipeline,
        reusable_protocols=reusable_protocols,
        capability_policy=capability_policy,
    )

    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "project_dir": review.get("project_dir") if review else None,
        "run_id": review.get("run_id") if review else None,
        "run_type": review.get("run_type") if review else None,
        "workflow_scope": workflow_scope,
        "run_summary": run_summary,
        "visual_reference": VISUAL_REFERENCE,
        "artifact_history": artifact_history,
        "reviewer_gate": reviewer_gate,
        "fig5_checklist": fig5_checklist,
        "discovery_pipeline": discovery_pipeline,
        "feature_alignment": feature_alignment,
        "reusable_protocols": reusable_protocols,
        "native_renderers": renderers,
        "capability_policy": capability_policy,
        "privacy": {
            "patient_rows_returned": False,
            "raw_datasets_returned": False,
            "external_image_loaded_by_api": False,
            "source": "local_agent_review_artifacts" if review else "empty_state",
        },
    }


def _artifact_history(
    *,
    artifacts: List[Dict[str, Any]],
    payloads: Dict[str, Dict[str, Any]],
    ledger: Dict[str, Any],
    project_dir: Optional[str],
) -> Dict[str, Any]:
    names = [str(a.get("name") or a.get("relative_path") or "") for a in artifacts]
    ledger_names = {
        str(a.get("name") or a.get("relative_path") or "")
        for a in ledger.get("artifacts", [])
        if isinstance(a, dict)
    }
    items = []
    for artifact in artifacts:
        name = str(artifact.get("name") or artifact.get("relative_path") or "")
        if not name:
            continue
        payload = payloads.get(name, {})
        items.append(
            {
                "name": name,
                "title": _artifact_title(name),
                "category": _artifact_category(name),
                "sha256": artifact.get("sha256"),
                "sha256_short": str(artifact.get("sha256") or "")[:12],
                "bytes": artifact.get("bytes"),
                "history_tabs": _history_tabs(name, payload),
                "code_refs": _code_refs(name),
                "inputs": _artifact_inputs(name, payload),
                "environment": _environment_for_artifact(name, payload, ledger),
                "review": _artifact_review(name, payload),
                "evidence_ids": _evidence_ids(name, payload),
                "in_ledger": name in ledger_names or not ledger_names,
                "provenance_complete": bool(artifact.get("sha256")) and (
                    name in ledger_names or not ledger_names
                ),
            }
        )
    return {
        "source": "agent_run_review" if project_dir else "empty_state",
        "project_dir": project_dir,
        "items": items,
        "count": len(items),
        "hashed_count": sum(1 for item in items if item.get("sha256")),
        "ledger_count": len(ledger_names) if ledger_names else len(items),
        "tabs": ["Code", "Execution Log", "Messages", "Environment", "Review"],
        "available_artifacts": names,
    }


def _reviewer_gate(
    *,
    review: Optional[Dict[str, Any]],
    payloads: Dict[str, Dict[str, Any]],
    gate: Dict[str, Any],
    readiness: Dict[str, Any],
    ledger: Dict[str, Any],
) -> Dict[str, Any]:
    draft = payloads.get("manuscript_draft.json", {})
    cohort = payloads.get("cohort_summary.json", {})
    figure = payloads.get("figure_gallery.json", {})
    checks = gate.get("checks", []) if isinstance(gate.get("checks"), list) else []

    strict = ledger.get("strict_evidence_audit") or {}
    numeric = ledger.get("numeric_evidence_audit") or {}
    privacy = (ledger.get("privacy") or {}).get("artifact_scan") or {}
    gate_map = {str(row.get("id")): row for row in checks if isinstance(row, dict)}
    claim_rows = _claim_rows(draft)

    reviewer_checks = [
        _check(
            "citation_fidelity",
            "Citation / claim fidelity",
            _claims_have_evidence(claim_rows),
            "manuscript_draft.json",
            "Every visible claim or sentence carries evidence IDs."
            if claim_rows
            else "No manuscript claim artifact is present in this run.",
            status_if_empty="not_applicable" if not claim_rows else None,
        ),
        _check(
            "numeric_traceability",
            "Numeric claim traceability",
            _numeric_passed(numeric),
            "evidence_ledger.json",
            "Numeric audit passed or no numeric manuscript claims were produced.",
            status_if_empty="not_applicable" if not numeric else None,
        ),
        _check(
            "figure_source_consistency",
            "Figure-source consistency",
            _figure_consistent(figure, payloads),
            "figure_gallery.json",
            "Figures are present only when a local figure/source-data artifact exists.",
            status_if_empty="not_applicable" if not figure else None,
        ),
        _check(
            "denominator_reporting",
            "Denominator reporting",
            _denominator_present(cohort, payloads),
            "cohort_summary.json",
            "Cohort summary or Table 1 reports a denominator before claims.",
        ),
        _check(
            "privacy_scan",
            "Privacy scan",
            bool(gate_map.get("no_patient_rows_persisted", {}).get("passed"))
            or bool(privacy.get("passed")),
            "quality_gate.json",
            "No patient rows are persisted in review artifacts.",
        ),
        _check(
            "conclusion_safety",
            "Conclusion safety",
            gate.get("reportable") is False and gate.get("draft_unlocked") is False,
            "quality_gate.json",
            "Exploratory or preflight outputs remain locked until explicit review.",
        ),
        _check(
            "human_signoff",
            "Human sign-off",
            bool(review and review.get("signed")),
            "human_signoff.json",
            "Local sign-off is recorded separately and never unlocks draft claims.",
            warning_ok=True,
        ),
    ]
    passed = sum(1 for row in reviewer_checks if row["status"] in {"passed", "not_applicable"})
    return {
        "status": readiness.get("status") or gate.get("status") or "not_started",
        "reason": gate.get("reason"),
        "reportable": bool(gate.get("reportable")),
        "draft_unlocked": bool(gate.get("draft_unlocked")),
        "signed": bool(review and review.get("signed")),
        "signable": bool(readiness.get("signable")),
        "passed_count": passed,
        "total_count": len(reviewer_checks),
        "checks": reviewer_checks,
        "automated_gate_checks": checks,
        "open_blockers": [
            row["id"]
            for row in reviewer_checks
            if row["status"] in {"failed", "needs_review"}
        ],
        "strict_evidence_audit": strict,
        "numeric_evidence_audit": numeric,
        "source": "quality_gate.json",
    }


def _workflow_scope(
    review: Optional[Dict[str, Any]], payloads: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    if not review:
        return {
            "id": "empty_state",
            "label": "No active run",
            "tone": "info",
            "supports_fig5": False,
            "supports_benchmark_review": False,
            "detail": "Open a local Agent run before reviewing evidence readiness.",
        }

    run_type = str(review.get("run_type") or "").lower()
    project_dir = str(review.get("project_dir") or "").lower()
    run_context = payloads.get("run_context.json", {})
    source = run_context.get("source") if isinstance(run_context, dict) else {}
    source = source if isinstance(source, dict) else {}
    source_text = " ".join(
        str(value or "").lower()
        for value in (
            source.get("label"),
            source.get("database"),
            run_context.get("study_id") if isinstance(run_context, dict) else "",
        )
    )

    if "canonical9" in run_type or "fig2" in source_text or "benchmark" in source_text:
        return {
            "id": "canonical_benchmark",
            "label": "Canonical benchmark evidence review",
            "tone": "info",
            "supports_fig5": False,
            "supports_benchmark_review": True,
            "detail": "Figure 2 capability evidence, not a Figure 5 discovery candidate.",
        }
    if (
        "idea" in run_type
        or "discovery" in run_type
        or "idea-" in project_dir
        or "discovery" in project_dir
    ):
        return {
            "id": "idea_discovery",
            "label": "Idea Mining discovery candidate",
            "tone": "ok",
            "supports_fig5": True,
            "supports_benchmark_review": False,
            "detail": "Figure 5 discovery candidate once source-data gates pass.",
        }
    return {
        "id": "agent_review",
        "label": "General Agent review",
        "tone": "warn",
        "supports_fig5": False,
        "supports_benchmark_review": False,
        "detail": "Useful for artifact review, but not yet classified as benchmark or discovery evidence.",
    }


def _run_summary(
    *,
    review: Optional[Dict[str, Any]],
    payloads: Dict[str, Dict[str, Any]],
    artifact_history: Dict[str, Any],
    reviewer_gate: Dict[str, Any],
    renderers: List[Dict[str, Any]],
    workflow_scope: Dict[str, Any],
) -> Dict[str, Any]:
    run_context = payloads.get("run_context.json", {})
    source = run_context.get("source") if isinstance(run_context, dict) else {}
    source = source if isinstance(source, dict) else {}
    denominator = _denominator_value(payloads)
    renderer_ready = sum(1 for row in renderers if row.get("can_render"))
    artifact_count = int(artifact_history.get("count") or 0)
    hashed_count = int(artifact_history.get("hashed_count") or 0)
    review_passed = int(reviewer_gate.get("passed_count") or 0)
    review_total = int(reviewer_gate.get("total_count") or 0)
    run_ctx = run_context if isinstance(run_context, dict) else {}
    # question is optional on /api/jobs/agent-run (preflight runs usually
    # have none) — fall back to the study/run identity so a loaded review
    # never renders the "No active local Agent run" empty-state header.
    title = (
        _short_text(run_ctx.get("question"))
        or _short_text(run_ctx.get("study_id"))
        or (f"Run {run_ctx.get('run_id')}" if run_ctx.get("run_id") else "")
        or "No active local Agent run"
    )
    source_label = _short_text(
        source.get("label")
        or source.get("database")
        or run_context.get("study_id")
        if isinstance(run_context, dict)
        else None
    )
    return {
        "title": title,
        "source_label": source_label or "Open a local run to populate source context",
        "database": source.get("database"),
        "denominator": denominator,
        "status": "reportable" if reviewer_gate.get("reportable") else "review_locked",
        "status_label": "Reportable" if reviewer_gate.get("reportable") else "Review locked",
        "workflow_scope": workflow_scope,
        "next_action": _next_action_for_gate(reviewer_gate),
        "kpis": [
            {
                "id": "artifacts",
                "label": "Artifacts",
                "value": artifact_count,
                "detail": f"{hashed_count} hashed",
            },
            {
                "id": "review_checks",
                "label": "Reviewer checks",
                "value": f"{review_passed}/{review_total}" if review_total else "0/0",
                "detail": "passed or not applicable",
            },
            {
                "id": "renderers",
                "label": "ICU renderers",
                "value": f"{renderer_ready}/{len(renderers)}",
                "detail": "ready",
            },
            {
                "id": "denominator",
                "label": "Denominator",
                "value": denominator if denominator is not None else "—",
                "detail": "aggregate only",
            },
        ],
        "local_only": {
            "patient_rows_returned": False,
            "raw_datasets_returned": False,
            "project_dir": review.get("project_dir") if review else None,
        },
    }


def _fig5_checklist(
    *,
    payloads: Dict[str, Dict[str, Any]],
    artifact_history: Dict[str, Any],
    reviewer_gate: Dict[str, Any],
    renderers: List[Dict[str, Any]],
    workflow_scope: Dict[str, Any],
) -> Dict[str, Any]:
    run_context = payloads.get("run_context.json", {})
    source = run_context.get("source") if isinstance(run_context, dict) else {}
    source = source if isinstance(source, dict) else {}
    question = run_context.get("question") if isinstance(run_context, dict) else None
    denominator = _denominator_value(payloads)
    artifact_count = int(artifact_history.get("count") or 0)
    hashed_count = int(artifact_history.get("hashed_count") or 0)
    ledger_count = int(artifact_history.get("ledger_count") or 0)
    ready_renderers = [row for row in renderers if row.get("can_render")]
    reviewer_blockers = list(reviewer_gate.get("open_blockers") or [])
    checks = reviewer_gate.get("checks") if isinstance(reviewer_gate.get("checks"), list) else []
    figure_status = _check_status(checks, "figure_source_consistency")
    supports_fig5 = bool(workflow_scope.get("supports_fig5"))
    checklist_title = (
        "Fig5 / Idea Mining readiness"
        if supports_fig5
        else "Evidence readiness checklist"
    )
    checklist_description = (
        "Checks whether a discovery candidate is ready for Figure 5 source-data freeze."
        if supports_fig5
        else str(workflow_scope.get("detail") or "Checks whether the current run has enough local evidence for review.")
    )
    items = [
        _checklist_item(
            "workflow_scope",
            "Workflow scope classified",
            "passed" if workflow_scope.get("id") != "empty_state" else "needs_review",
            "run_context.json",
            str(workflow_scope.get("detail") or "Classify this run before using downstream readiness gates."),
            "run_context.json",
        ),
        _checklist_item(
            "source_signal",
            "Literature or source signal captured",
            (
                "passed"
                if supports_fig5 and question and source
                else "not_applicable"
                if not supports_fig5
                else "needs_review"
            ),
            "run_context.json",
            "Lock the citation, quote hash, and source scope before treating this as a discovery vignette."
            if supports_fig5
            else "Not required for canonical benchmark evidence review.",
            "run_context.json",
        ),
        _checklist_item(
            "outcome_blind_feasibility",
            "Outcome-blind feasibility gate",
            "passed" if denominator is not None else "needs_review",
            "cohort_summary.json",
            "Resolve denominator, database scope, concepts, and time window before looking at effects.",
            "cohort_summary.json",
        ),
        _checklist_item(
            "artifact_provenance",
            "Artifact provenance envelope",
            "passed"
            if artifact_count > 0 and hashed_count == artifact_count and ledger_count >= artifact_count
            else "needs_review",
            "evidence_ledger.json",
            "Every review artifact should have a hash and ledger entry before figure drafting.",
            "evidence_ledger.json",
        ),
        _checklist_item(
            "reviewer_gate",
            "Reviewer checks clear enough for draft use",
            "passed" if not reviewer_blockers and reviewer_gate.get("signed") else "needs_review",
            "quality_gate.json",
            "Clear failed checks and record human sign-off before any manuscript claim is unlocked.",
            "quality_gate.json",
        ),
        _checklist_item(
            "native_renderers",
            "ICU-native renderers ready",
            "passed" if len(ready_renderers) >= 3 else "needs_review",
            "quality_gate.json",
            "Use concept, cohort, time-lane, and claim graph previews to check clinical coherence.",
            "quality_gate.json",
        ),
        _checklist_item(
            "figure_source_data",
            "Figure and source-data boundary",
            "not_applicable"
            if figure_status == "not_applicable"
            else ("passed" if figure_status == "passed" else "needs_review"),
            "figure_gallery.json",
            "Generate code-backed source data before freezing any Fig5 result panel.",
            "figure_gallery.json",
        ),
    ]
    applicable = [row for row in items if row["status"] != "not_applicable"]
    passed = sum(1 for row in applicable if row["status"] == "passed")
    return {
        "title": checklist_title,
        "description": checklist_description,
        "candidate_for_fig5": supports_fig5,
        "workflow_scope": workflow_scope,
        "progress": round(passed / len(applicable), 3) if applicable else 0.0,
        "passed_count": passed,
        "applicable_count": len(applicable),
        "items": items,
        "next_action": next(
            (row["next_action"] for row in items if row["status"] == "needs_review"),
            "Checklist is ready for human review and figure-source-data freeze.",
        ),
    }


def _alignment_item(
    item_id: str,
    label: str,
    status: str,
    evidence: str,
    focus_artifact: Optional[str],
) -> Dict[str, Any]:
    return {
        "id": item_id,
        "label": label,
        "status": status,
        "status_label": _status_label(status),
        "evidence": evidence,
        "focus_artifact": focus_artifact,
    }


def _alignment_stage_status(stage: Dict[str, Any], has_run: bool) -> str:
    if not has_run:
        return "waiting_for_run"
    status = str(stage.get("status") or "")
    if status in {"passed", "needs_review", "failed", "not_applicable"}:
        return status
    return "needs_review"


def _status_label(status: str) -> str:
    return {
        "passed": "passed / 已通过",
        "needs_review": "needs review / 待审阅",
        "failed": "failed / 未通过",
        "waiting_for_run": "waiting for run / 等待运行",
        "not_applicable": "not applicable / 不适用",
        "unavailable": "unavailable / 不可用",
    }.get(str(status or ""), str(status or "unknown"))


def _feature_alignment(
    *,
    artifact_history: Dict[str, Any],
    reviewer_gate: Dict[str, Any],
    renderers: List[Dict[str, Any]],
    workflow_scope: Dict[str, Any],
    discovery_pipeline: Dict[str, Any],
    reusable_protocols: List[Dict[str, Any]],
    capability_policy: Dict[str, Any],
) -> Dict[str, Any]:
    artifact_count = int(artifact_history.get("count") or 0)
    reviewer_total = int(reviewer_gate.get("total_count") or 0)
    renderer_ready = sum(1 for row in renderers if row.get("can_render"))
    discovery_stages = {
        str(row.get("id")): row
        for row in discovery_pipeline.get("stages", [])
        if isinstance(row, dict)
    }
    prior_art_stage = discovery_stages.get("prior_art") or {}
    feasibility_stage = discovery_stages.get("outcome_blind_feasibility") or {}
    discovery_has_run = bool(discovery_pipeline.get("latest_run_id"))
    capability_flags = capability_policy.get("settings") or {}
    skills_enabled = bool(capability_flags.get("science_skills_enabled", True))
    return {
        "title": "Workbench coverage / 工作台覆盖",
        "source": "public_claude_science_reference",
        "workflow_scope": workflow_scope,
        "items": [
            _alignment_item(
                item_id="artifact_history",
                label="Artifact history / 产物历史",
                status="passed" if artifact_count else "waiting_for_run",
                evidence=f"{artifact_count} local artifacts / 本地产物",
                focus_artifact="evidence_ledger.json",
            ),
            _alignment_item(
                item_id="reviewer_gate",
                label="Reviewer checks / 审阅检查",
                status=(
                    "passed"
                    if reviewer_total and reviewer_gate.get("reportable")
                    else ("needs_review" if reviewer_total else "waiting_for_run")
                ),
                evidence=f"{reviewer_gate.get('passed_count', 0)}/{reviewer_total} checks clear / 检查通过",
                focus_artifact="quality_gate.json",
            ),
            _alignment_item(
                item_id="reusable_protocols",
                label="Reusable ICU protocols / 可复用 ICU protocol",
                status="passed" if reusable_protocols else "unavailable",
                evidence=(
                    f"{len(reusable_protocols)} local protocol cards / 本地 protocol 卡片"
                    if skills_enabled
                    else "Skills disabled in Settings / Settings 已关闭 Skills"
                ),
                focus_artifact=None,
            ),
            _alignment_item(
                item_id="prior_art_gate",
                label="Prior-art review / 既有研究审阅",
                status=_alignment_stage_status(prior_art_stage, discovery_has_run),
                evidence=str(prior_art_stage.get("evidence") or "waiting for idea run"),
                focus_artifact=None,
            ),
            _alignment_item(
                item_id="outcome_blind_feasibility",
                label="Outcome-blind feasibility / 盲结局可行性",
                status=_alignment_stage_status(feasibility_stage, discovery_has_run),
                evidence=str(
                    feasibility_stage.get("evidence") or "waiting for active export"
                ),
                focus_artifact=None,
            ),
            _alignment_item(
                item_id="icu_native_renderers",
                label="ICU-native previews / ICU 原生预览",
                status="passed" if renderer_ready else "waiting_for_run",
                evidence=f"{renderer_ready}/{len(renderers)} renderers ready / 预览可用",
                focus_artifact="cohort_summary.json",
            ),
            _alignment_item(
                item_id="workflow_scope",
                label="Workflow role / 工作流角色",
                status="passed",
                evidence=str(workflow_scope.get("label") or "unclassified"),
                focus_artifact="run_context.json",
            ),
        ],
    }


def _discovery_pipeline() -> Dict[str, Any]:
    """Summarize the latest local Idea Mining run without executing it."""
    try:
        runs = idea_mining_web.list_runs({"limit": 1}).get("runs") or []
    except Exception:
        return _discovery_unavailable(
            latest_run_id=None,
            title="Idea Mining storage unavailable / Idea Mining 存储不可用",
        )
    if not runs:
        return {
            "status": "waiting_for_idea_run",
            "status_label": "Waiting for Idea Mining run / 等待 Idea Mining run",
            "latest_run_id": None,
            "title": "No local discovery candidate yet / 暂无本地发现候选",
            "source": None,
            "fig5_candidate_ready": False,
            "source_data_review_ready": False,
            "stages": _discovery_empty_stages(),
            "privacy": {
                "patient_rows_returned": False,
                "network_calls": 0,
                "external_llm_calls": 0,
            },
        }

    run_id = str(runs[0].get("run_id") or "").strip()
    try:
        payload = idea_mining_web.get_run({"run_id": run_id})
    except Exception:
        return _discovery_unavailable(
            latest_run_id=run_id or None,
            title=str(runs[0].get("title") or "Idea Mining run could not be loaded / Idea Mining run 无法读取"),
        )

    source = ((payload.get("source_evidence") or [{}])[0]) or {}
    idea = ((payload.get("idea_ledger") or [{}])[0]) or {}
    pre = payload.get("pre_experiment") or {}
    prior_art = _pipeline_prior_art(payload, idea)
    plan = payload.get("idea_plan") or {}
    handoff = payload.get("handoff") or {}
    project = payload.get("agent_project") or {}
    stages = [
        _pipeline_stage(
            "source_signal",
            "Source signal / 来源线索",
            "passed" if source.get("title") and source.get("evidence_quote") else "needs_review",
            _source_label(source),
            "Metadata, hash, and bounded quote are available; full text is not stored. / 已有元数据、hash 和有界引文；不保存全文。",
        ),
        _pipeline_stage(
            "prior_art",
            "Prior-art review / 既有研究审阅",
            (
                "passed"
                if prior_art.get("search_performed")
                and str(prior_art.get("status") or "") != "search_failed"
                else "needs_review"
            ),
            str(prior_art.get("status") or "not_checked"),
            str(
                prior_art.get("reason")
                or "Run bounded prior-art review before novelty claims. / 在新颖性表述前完成有界 prior-art 审阅。"
            ),
        ),
        _pipeline_stage(
            "outcome_blind_feasibility",
            "Outcome-blind feasibility / 盲结局可行性",
            "passed" if str(pre.get("status") or "") == "ready" else "needs_review",
            _feasibility_label(pre, idea),
            "Uses aggregate feature availability, denominator, and missingness before effect estimates. / 只看聚合特征可用性、分母和缺失结构，不先看效应估计。",
        ),
        _pipeline_stage(
            "go_no_go",
            "Go/no-go decision / 去留决策",
            "passed" if _go_decision_ready(idea) else "needs_review",
            _go_label(idea),
            "Only recommend/go candidates can enter Agent source-data review. / 只有 recommend/go 候选可进入 Agent 源数据审阅。",
        ),
        _pipeline_stage(
            "plan_replan",
            "Plan/replan / 计划与复审",
            "passed" if plan else "needs_review",
            _plan_label(plan, payload),
            "Plan review remains metadata-only and requires human confirmation before Agent execution. / 计划审阅只保存元数据，Agent 执行前仍需人工确认。",
        ),
        _pipeline_stage(
            "agent_handoff",
            "Agent handoff / Agent 交接",
            "passed" if handoff or project else "needs_review",
            _handoff_label(handoff, project),
            "Agent project seeds stay non-reportable until a real Agent run produces evidence IDs. / 只有真实 Agent run 产出 evidence IDs 后才可进入可报告审阅。",
        ),
    ]
    ready = all(row["status"] == "passed" for row in stages)
    privacy = _merged_pipeline_privacy(payload, prior_art)
    return {
        "status": "ready_for_agent_source_data_review" if ready else "needs_review",
        "status_label": (
            "Ready for Agent source-data review / 可进入 Agent 源数据审阅"
            if ready
            else "Needs review / 待审阅"
        ),
        "latest_run_id": run_id or None,
        "title": idea.get("idea_title") or runs[0].get("title") or "Discovery candidate",
        "go_no_go": idea.get("go_no_go"),
        "go_no_go_reason": idea.get("go_no_go_reason"),
        "source": {
            "title": source.get("title"),
            "journal": source.get("journal"),
            "year": source.get("year"),
            "doi": source.get("doi"),
            "pmid": source.get("pmid"),
        },
        "fig5_candidate_ready": ready,
        "mapped_concept_count": len(idea.get("mapped_concepts") or []),
        "feature_stat_count": len(pre.get("feature_statistics") or []),
        "cohort_entities": (pre.get("cohort") or {}).get("entities"),
        "stages": stages,
        "next_action": next(
            (row["next_action"] for row in stages if row["status"] == "needs_review"),
            "Run or attach an evidence-producing Agent analysis before freezing Figure 5 source data. / 冻结 Figure 5 source data 前，先运行或绑定能产出 evidence IDs 的 Agent 分析。",
        ),
        "privacy": {
            "patient_rows_returned": bool(privacy.get("patient_rows_returned")),
            "source_text_stored": bool(privacy.get("source_text_stored")),
            "network_calls": privacy["network_calls"],
            "external_llm_calls": privacy["external_llm_calls"],
        },
        "source_data_review_ready": ready,
    }


def _discovery_empty_stages() -> List[Dict[str, Any]]:
    return [
        _pipeline_stage(
            "source_signal",
            "Source signal / 来源线索",
            "needs_review",
            "not started / 未开始",
            "Run Idea Mining from a paper, PDF, review topic, or source quote. / 从论文、PDF、review topic 或来源引文启动 Idea Mining。",
        ),
        _pipeline_stage(
            "prior_art",
            "Prior-art review / 既有研究审阅",
            "needs_review",
            "not started / 未开始",
            "Bounded prior-art review is explicit opt-in and makes no network call until enabled. / 有界 prior-art 审阅必须显式 opt-in，启用前不会联网。",
        ),
        _pipeline_stage(
            "outcome_blind_feasibility",
            "Outcome-blind feasibility / 盲结局可行性",
            "needs_review",
            "not started / 未开始",
            "Register a local EasyICU export and run aggregate feasibility before any effect estimate. / 先注册本地 EasyICU export 并做聚合可行性检查，再看效应估计。",
        ),
        _pipeline_stage(
            "go_no_go",
            "Go/no-go decision / 去留决策",
            "needs_review",
            "not recorded / 未记录",
            "Only recommend/go candidates can enter Agent source-data review. / 只有 recommend/go 候选可进入 Agent 源数据审阅。",
        ),
        _pipeline_stage(
            "plan_replan",
            "Plan/replan / 计划与复审",
            "needs_review",
            "not planned / 未计划",
            "Review the study plan before creating an Agent handoff. / 创建 Agent 交接前先审阅研究计划。",
        ),
        _pipeline_stage(
            "agent_handoff",
            "Agent handoff / Agent 交接",
            "needs_review",
            "locked / 锁定",
            "Create a metadata-only Agent seed only after confirmation. / 人工确认后再创建仅含元数据的 Agent seed。",
        ),
    ]


def _discovery_unavailable(
    *, latest_run_id: Optional[str], title: str
) -> Dict[str, Any]:
    return {
        "status": "idea_mining_unavailable",
        "status_label": "Idea Mining unavailable / Idea Mining 不可用",
        "latest_run_id": latest_run_id,
        "title": title,
        "source": None,
        "fig5_candidate_ready": False,
        "source_data_review_ready": False,
        "stages": _discovery_empty_stages(),
        "next_action": (
            "Check the local Idea Mining storage before using this candidate. / "
            "先检查本地 Idea Mining 存储，再使用这个候选。"
        ),
        "privacy": {
            "patient_rows_returned": False,
            "source_text_stored": False,
            "network_calls": 0,
            "external_llm_calls": 0,
        },
    }


def _pipeline_stage(
    stage_id: str,
    label: str,
    status: str,
    evidence: str,
    next_action: str,
) -> Dict[str, Any]:
    return {
        "id": stage_id,
        "label": label,
        "status": status,
        "evidence": evidence,
        "next_action": next_action,
    }


def _pipeline_prior_art(payload: Dict[str, Any], idea: Dict[str, Any]) -> Dict[str, Any]:
    prior_art_check = payload.get("prior_art_check") or {}
    prior = prior_art_check.get("prior_art") if isinstance(prior_art_check, dict) else {}
    if isinstance(prior, dict) and prior:
        return prior
    prior = payload.get("prior_art") if isinstance(payload.get("prior_art"), dict) else {}
    if prior:
        return prior
    return idea.get("prior_art") if isinstance(idea.get("prior_art"), dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _merged_pipeline_privacy(
    payload: Dict[str, Any], prior_art: Dict[str, Any]
) -> Dict[str, Any]:
    privacy = payload.get("privacy") if isinstance(payload.get("privacy"), dict) else {}
    prior_art_check = (
        payload.get("prior_art_check")
        if isinstance(payload.get("prior_art_check"), dict)
        else {}
    )
    prior_privacy = (
        prior_art_check.get("privacy")
        if isinstance(prior_art_check.get("privacy"), dict)
        else {}
    )
    prior_network_calls = max(
        _safe_int(prior_privacy.get("network_calls")),
        _safe_int(prior_art.get("network_calls")),
    )
    prior_llm_calls = max(
        _safe_int(prior_privacy.get("external_llm_calls")),
        _safe_int(prior_art.get("external_llm_calls")),
    )
    return {
        "patient_rows_returned": bool(
            privacy.get("patient_rows_returned")
            or prior_privacy.get("patient_rows_returned")
        ),
        "source_text_stored": bool(
            privacy.get("source_text_stored") or prior_privacy.get("source_text_stored")
        ),
        "network_calls": _safe_int(privacy.get("network_calls")) + prior_network_calls,
        "external_llm_calls": _safe_int(privacy.get("external_llm_calls"))
        + prior_llm_calls,
    }


def _go_decision_ready(idea: Dict[str, Any]) -> bool:
    decision = str(idea.get("go_no_go") or "").strip().lower()
    return decision in {"recommend", "go"}


def _go_label(idea: Dict[str, Any]) -> str:
    decision = str(idea.get("go_no_go") or "").strip()
    reason = str(idea.get("go_no_go_reason") or "").strip()
    if not decision:
        return "not recorded / 未记录"
    return f"{decision} · {reason}" if reason else decision


def _source_label(source: Dict[str, Any]) -> str:
    bits = [
        str(source.get("title") or "source"),
        str(source.get("journal") or ""),
        str(source.get("year") or ""),
    ]
    return " · ".join(bit for bit in bits if bit)


def _feasibility_label(pre: Dict[str, Any], idea: Dict[str, Any]) -> str:
    status = str(pre.get("status") or "not_available")
    feasibility = idea.get("feasibility") if isinstance(idea.get("feasibility"), dict) else {}
    label = feasibility.get("label") or feasibility.get("tier")
    entities = (pre.get("cohort") or {}).get("entities")
    if entities is not None:
        return f"{status} · n={entities}"
    return f"{status} · {label}" if label else status


def _plan_label(plan: Dict[str, Any], payload: Dict[str, Any]) -> str:
    if plan:
        return str((plan.get("plan") or {}).get("plan_status") or plan.get("mode") or "planned")
    handoff_plan = payload.get("handoff_plan") if isinstance(payload.get("handoff_plan"), dict) else {}
    steps = handoff_plan.get("analysis_plan") if isinstance(handoff_plan, dict) else []
    return f"draft handoff plan · {len(steps or [])} steps" if steps else "not planned"


def _handoff_label(handoff: Dict[str, Any], project: Dict[str, Any]) -> str:
    if project:
        return str(project.get("study_id") or project.get("status") or "project seeded")
    if handoff:
        return str(handoff.get("idea_id") or handoff.get("candidate_topic") or "handoff frozen")
    return "locked until plan confirmation"


def _checklist_item(
    item_id: str,
    label: str,
    status: str,
    evidence: str,
    next_action: str,
    focus_artifact: str,
) -> Dict[str, Any]:
    return {
        "id": item_id,
        "label": label,
        "status": status,
        "evidence": evidence,
        "next_action": next_action,
        "focus_artifact": focus_artifact,
    }


def _check_status(checks: List[Dict[str, Any]], check_id: str) -> str:
    for row in checks:
        if isinstance(row, dict) and row.get("id") == check_id:
            return str(row.get("status") or "needs_review")
    return "needs_review"


def _denominator_value(payloads: Dict[str, Dict[str, Any]]) -> Optional[Any]:
    cohort = payloads.get("cohort_summary.json", {})
    table1 = payloads.get("table1_summary.json", {})
    summary = cohort.get("summary") if isinstance(cohort, dict) else {}
    for value in (
        (summary or {}).get("stays"),
        cohort.get("denominator") if isinstance(cohort, dict) else None,
        table1.get("denominator") if isinstance(table1, dict) else None,
        ((table1.get("source") or {}).get("stays") if isinstance(table1, dict) else None),
    ):
        if value is not None:
            return value
    return None


def _next_action_for_gate(reviewer_gate: Dict[str, Any]) -> str:
    blockers = list(reviewer_gate.get("open_blockers") or [])
    if reviewer_gate.get("reportable"):
        return "Freeze source data and prepare manuscript-facing figure panels."
    if blockers:
        return f"Resolve reviewer gate blocker: {blockers[0]}."
    if not reviewer_gate.get("signed"):
        return "Record human sign-off after reviewing local artifacts."
    return "Keep draft locked until the manuscript section has source-data review."


def _short_text(value: Any, limit: int = 180) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.split("\n\n", 1)[0].replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _native_renderers(
    payloads: Dict[str, Dict[str, Any]],
    artifacts: List[Dict[str, Any]],
    gate: Dict[str, Any],
) -> List[Dict[str, Any]]:
    names = {str(a.get("name") or a.get("relative_path") or "") for a in artifacts}
    rows = []
    for renderer in NATIVE_RENDERERS:
        wanted = list(renderer["artifact_names"])
        available = [name for name in wanted if name in names or name in payloads]
        rows.append(
            {
                **renderer,
                "available_artifacts": available,
                "can_render": bool(available),
                "preview": _renderer_preview(renderer["id"], payloads, gate),
            }
        )
    return rows


def _renderer_preview(
    renderer_id: str,
    payloads: Dict[str, Dict[str, Any]],
    gate: Dict[str, Any],
) -> Dict[str, Any]:
    if renderer_id == "concept_coverage_matrix":
        quality = payloads.get("quality_gate.json", {}).get("quality") or []
        # Both missingness payload builders emit the feature list under
        # "rows" (agent_outputs metadata + full paths); there is no
        # "features" key, so that lookup left this fallback permanently dead.
        missing = payloads.get("missingness_audit.json", {}).get("rows") or []
        return {
            "rows": [
                {
                    "label": str(row.get("module") or row.get("feature") or "module"),
                    "value": row.get("coverage_pct"),
                    "status": row.get("status") or "review",
                }
                for row in (quality or missing)[:8]
                if isinstance(row, dict)
            ]
        }
    if renderer_id == "cohort_attrition":
        cohort = payloads.get("cohort_summary.json", {})
        table1 = payloads.get("table1_summary.json", {})
        summary = cohort.get("summary") or table1.get("source") or {}
        groups = table1.get("groups") if isinstance(table1.get("groups"), list) else []
        return {
            "denominator": summary.get("stays") or table1.get("denominator"),
            "groups": groups[:4],
        }
    if renderer_id == "icu_time_lane":
        checks = gate.get("checks") if isinstance(gate.get("checks"), list) else []
        return {
            "events": [
                {
                    "label": row.get("label") or row.get("id"),
                    "status": "passed" if row.get("passed") else "review",
                }
                for row in checks[:6]
                if isinstance(row, dict)
            ]
        }
    if renderer_id == "claim_evidence_graph":
        draft = payloads.get("manuscript_draft.json", {})
        claims = _claim_rows(draft)
        return {
            "claims": [
                {
                    "label": row.get("claim_id")
                    or row.get("sentence_id")
                    or f"claim_{idx + 1:02d}",
                    "evidence_ids": row.get("evidence_ids") or [],
                }
                for idx, row in enumerate(claims[:6])
                if isinstance(row, dict)
            ]
        }
    return {}


def _check(
    check_id: str,
    label: str,
    passed: bool,
    evidence: str,
    detail: str,
    *,
    status_if_empty: Optional[str] = None,
    warning_ok: bool = False,
) -> Dict[str, Any]:
    if status_if_empty:
        status = status_if_empty
    elif passed:
        status = "passed"
    else:
        status = "needs_review" if warning_ok else "failed"
    return {
        "id": check_id,
        "label": label,
        "status": status,
        "passed": bool(passed),
        "evidence": evidence,
        "detail": detail,
    }


def _claims_have_evidence(claims: List[Dict[str, Any]]) -> bool:
    if not claims:
        return True
    return all(bool(row.get("evidence_ids")) for row in claims if isinstance(row, dict))


def _claim_rows(draft: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims = draft.get("claims") if isinstance(draft, dict) else []
    sentences = draft.get("sentences") if isinstance(draft, dict) else []
    rows = []
    if isinstance(claims, list):
        rows.extend(row for row in claims if isinstance(row, dict))
    if isinstance(sentences, list):
        rows.extend(row for row in sentences if isinstance(row, dict))
    return rows


def _numeric_passed(numeric: Dict[str, Any]) -> bool:
    if not numeric:
        return True
    return bool(numeric.get("passed")) or int(numeric.get("numeric_mention_count") or 0) == 0


def _figure_consistent(
    figure: Dict[str, Any], payloads: Dict[str, Dict[str, Any]]
) -> bool:
    if not figure:
        return True
    rows = figure.get("figures") if isinstance(figure.get("figures"), list) else []
    if not rows:
        # A gallery payload with no figure entries has nothing to check.
        return True
    # evidence_ledger.json must NOT satisfy this check: read_run_review
    # refuses run dirs without it, so using it as a fallback made the check
    # unconditionally pass. Figures count as source-consistent only when the
    # run manifest is present and every gallery entry points back to its
    # figure contract.
    if not payloads.get("source_run_manifest.json"):
        return False
    return all(
        str(row.get("contract_path") or "").strip()
        for row in rows
        if isinstance(row, dict)
    )


def _denominator_present(
    cohort: Dict[str, Any], payloads: Dict[str, Dict[str, Any]]
) -> bool:
    table1 = payloads.get("table1_summary.json", {})
    summary = cohort.get("summary") if isinstance(cohort, dict) else {}
    return bool(
        (summary or {}).get("stays")
        or cohort.get("denominator")
        or table1.get("denominator")
        or (table1.get("source") or {}).get("stays")
    )


def _history_tabs(name: str, payload: Dict[str, Any]) -> Dict[str, str]:
    return {
        "Code": "; ".join(_code_refs(name)),
        "Execution Log": _execution_log(name, payload),
        "Messages": _message_summary(name, payload),
        "Environment": _environment_for_artifact(name, payload, {}),
        "Review": _artifact_review(name, payload),
    }


def _code_refs(name: str) -> List[str]:
    if name in {"table1_summary.json", "missingness_audit.json", "roc_curve.json", "calibration_curve.json"}:
        return ["easyicu.webserver.agent_outputs.build_agent_output_artifacts"]
    if name in {"quality_gate.json", "evidence_ledger.json", "human_signoff.json"}:
        return ["easyicu.webserver.agent_runs._evaluate_gate_with_ledger"]
    if name in {"agent_plan.json", "manuscript_draft.json"}:
        return ["easyicu.webserver.agent_runs.make_agent_run_runner"]
    if name in {"benchmark_scorecard.json", "workflow_graph.json", "figure_gallery.json", "source_run_manifest.json"}:
        return ["canonical9 import/review package"]
    return ["easyicu.webserver.agent_runs.make_agent_run_runner"]


def _artifact_inputs(name: str, payload: Dict[str, Any]) -> List[str]:
    if name == "run_context.json":
        return ["active export registry", "study id", "question", "local-first settings"]
    if name == "cohort_summary.json":
        return ["workspace summary", "cohort counts"]
    if name == "quality_gate.json":
        return ["source summary", "quality rows", "privacy scan", "strict/numeric audits"]
    if name == "evidence_ledger.json":
        return ["artifact registry", "quality gate", "provider boundary", "privacy scan"]
    if name == "human_signoff.json":
        return ["reviewer confirmations", "current artifact hashes", "readiness gate"]
    if "figure" in name:
        return ["local figure artifact", "source data metadata"]
    if isinstance(payload, dict) and payload:
        return sorted(str(key) for key in list(payload.keys())[:4])
    return ["local Agent run artifact"]


def _environment_for_artifact(
    name: str, payload: Dict[str, Any], ledger: Dict[str, Any]
) -> str:
    provider = ledger.get("provider") if isinstance(ledger, dict) else {}
    if name in {"agent_plan.json", "manuscript_draft.json"} and provider:
        return str(provider.get("provider") or provider.get("mode") or "agent provider")
    local_first = payload.get("local_first") if isinstance(payload, dict) else None
    if isinstance(local_first, dict):
        return f"local FastAPI runtime · uploads={local_first.get('uploads', 0)} · tokens={local_first.get('tokens', 0)}"
    return "local FastAPI runtime · bounded JSON artifacts"


def _artifact_review(name: str, payload: Dict[str, Any]) -> str:
    if name == "quality_gate.json":
        gate = payload.get("gate") if isinstance(payload, dict) else {}
        return str((gate or {}).get("reason") or (gate or {}).get("status") or "quality gate")
    if name == "evidence_ledger.json":
        privacy = payload.get("privacy") if isinstance(payload, dict) else {}
        scan = (privacy or {}).get("artifact_scan") or {}
        return "privacy scan passed" if scan.get("passed") else "review ledger before reporting"
    if name == "human_signoff.json":
        return "local sign-off recorded; reportable remains false"
    if name == "manuscript_draft.json":
        return "draft claims remain locked until evidence and human review pass"
    return "available for reviewer inspection"


def _execution_log(name: str, payload: Dict[str, Any]) -> str:
    if name == "run_context.json":
        return "source registry resolved; export snapshot summarized"
    if name == "quality_gate.json":
        gate = payload.get("gate") if isinstance(payload, dict) else {}
        return f"gate evaluated: {(gate or {}).get('status', 'unknown')}"
    if name == "evidence_ledger.json":
        artifacts = payload.get("artifacts") if isinstance(payload, dict) else []
        return f"registered {len(artifacts) if isinstance(artifacts, list) else 0} artifacts"
    return "written by local Agent review pipeline"


def _message_summary(name: str, payload: Dict[str, Any]) -> str:
    if name == "manuscript_draft.json":
        claims = _claim_rows(payload)
        return f"{len(claims)} draft claim/sentence rows bound for review"
    if name == "agent_plan.json":
        steps = payload.get("steps") if isinstance(payload, dict) else []
        return f"{len(steps) if isinstance(steps, list) else 0} planned Agent steps"
    return "bounded artifact message history proxy"


def _evidence_ids(name: str, payload: Dict[str, Any]) -> List[str]:
    ids = {name}
    for row in _claim_rows(payload if isinstance(payload, dict) else {}):
        for evidence_id in row.get("evidence_ids") or []:
            ids.add(str(evidence_id))
    return sorted(ids)


def _artifact_title(name: str) -> str:
    labels = {
        "run_context.json": "Run context",
        "cohort_summary.json": "Cohort summary",
        "table1_summary.json": "Table 1 summary",
        "missingness_audit.json": "Missingness audit",
        "roc_curve.json": "ROC curve",
        "calibration_curve.json": "Calibration curve",
        "quality_gate.json": "Quality gate",
        "evidence_ledger.json": "Evidence ledger",
        "agent_plan.json": "Agent plan",
        "manuscript_draft.json": "Locked manuscript draft",
        "benchmark_scorecard.json": "Benchmark scorecard",
        "workflow_graph.json": "Workflow graph",
        "figure_gallery.json": "Figure gallery",
        "source_run_manifest.json": "Source run manifest",
        "human_signoff.json": "Human sign-off",
    }
    return labels.get(name, name.replace("_", " ").replace(".json", "").title())


def _artifact_category(name: str) -> str:
    if "figure" in name:
        return "figure"
    if "ledger" in name or "gate" in name or "signoff" in name:
        return "review"
    if "draft" in name or "plan" in name:
        return "writing"
    if "cohort" in name or "table1" in name or "missingness" in name:
        return "analysis"
    if "manifest" in name or "context" in name:
        return "provenance"
    return "artifact"
