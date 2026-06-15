"""Pure UI snapshot adapters shared by the EasyICU web surfaces.

These adapters intentionally read only plain mappings and return immutable
data structures. Rendering stays in Streamlit modules; cohort filtering,
concept selection, manifest loading, and LLM routing remain owned by the
existing classic/Copilot/Agent engines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_COPILOT_STEP_ORDER = ("question", "data", "cohort", "concepts", "extract", "review", "analysis", "draft")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _len_or_zero(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    return 0


@dataclass(frozen=True)
class StudyWorkspaceSnapshot:
    """Render-ready status model for Copilot and classic study state."""

    active_step: str
    active_step_label: str
    current_decision: str
    data_label: str
    cohort_label: str
    concepts_label: str
    review_label: str
    agent_label: str
    selected_concepts: int
    loaded_concepts: int
    patient_count: int
    path_validated: bool
    step_done: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentProjectSnapshot:
    """Render-ready status model for the Agent Projects workspace."""

    project_id: str
    project_title: str
    active_view: str
    status: str
    status_tone: str
    run_id: str
    run_dir: str
    question: str
    cohort_label: str
    step_total: int
    step_ok: int
    step_failed: int
    evidence_count: int
    figure_count: int
    table_count: int
    finding_errors: int
    finding_warnings: int
    gates_total: int
    gates_blocked: int
    review_decision: str
    history_count: int
    history_runs: tuple[dict[str, str], ...] = field(default_factory=tuple)


def _step_index(step: str) -> int:
    try:
        return _COPILOT_STEP_ORDER.index(step)
    except ValueError:
        return 0


def build_study_workspace_snapshot(
    state: Mapping[str, Any],
    study: Mapping[str, Any] | None = None,
    *,
    lang: str = "en",
) -> StudyWorkspaceSnapshot:
    """Build the Copilot/classic shared study snapshot without side effects."""
    is_en = lang == "en"
    study = study or {}
    entry_mode = str(state.get("entry_mode") or study.get("data_mode") or "none")
    database = str(state.get("database") or "miiv")
    db_label = "mock" if database == "mock" else database.upper()
    data_path = str(state.get("data_path") or "").strip()
    path_validated = bool(state.get("path_validated"))
    selected_count = _len_or_zero(state.get("selected_concepts"))
    loaded_count = _len_or_zero(state.get("loaded_concepts"))
    patient_count = _len_or_zero(state.get("patient_ids"))
    active_step = str(study.get("step") or "question")
    if active_step not in _COPILOT_STEP_ORDER:
        active_step = "question"
    active_idx = _step_index(active_step)

    mode_label = {
        "demo": "Demo" if is_en else "演示",
        "real": "Real Data" if is_en else "真实数据",
        "none": "Not selected" if is_en else "未选择",
    }.get(entry_mode, entry_mode)
    data_label = f"{mode_label} · {db_label}"
    if data_path:
        data_label += f" · {Path(data_path).name}"

    step_done = {
        "question": bool(state.get("research_agent_question") or str(study.get("question") or "").strip()),
        "data": entry_mode == "demo" or bool(data_path and path_validated) or active_idx >= _step_index("data"),
        "cohort": bool(state.get("step2_confirmed")) or active_idx >= _step_index("cohort"),
        "concepts": bool(state.get("step3_confirmed")) or selected_count > 0 or active_idx >= _step_index("concepts"),
        "review": loaded_count > 0 or patient_count > 0 or active_idx >= _step_index("review"),
    }
    step_done["analysis"] = bool(state.get("research_agent_question")) or step_done["review"] or active_idx >= _step_index("analysis")
    step_done["draft"] = active_idx >= _step_index("draft")

    labels_en = {
        "question": "Question",
        "data": "Data source",
        "cohort": "Cohort",
        "concepts": "Concept modules",
        "extract": "Extraction",
        "review": "Review",
        "analysis": "Agent handoff",
        "draft": "Draft gate",
    }
    labels_zh = {
        "question": "研究问题",
        "data": "数据源",
        "cohort": "队列",
        "concepts": "变量模块",
        "extract": "数据抽取",
        "review": "审阅",
        "analysis": "Agent 交接",
        "draft": "草稿闸门",
    }
    labels = labels_en if is_en else labels_zh

    decision_by_step = {
        "question": "Frame one study question" if is_en else "先确定一个研究问题",
        "data": "Choose or validate the local data path" if is_en else "选择或验证本地数据路径",
        "cohort": "Confirm the eligible ICU cohort" if is_en else "确认纳入 ICU 队列",
        "concepts": "Keep core modules or add optional modules" if is_en else "保留核心模块，或添加可选模块",
        "extract": "Run deterministic local extraction" if is_en else "运行确定性本地抽取",
        "review": "Review overview and top quality issues" if is_en else "审阅概览与主要质量问题",
        "analysis": "Hand the study to Agent Projects" if is_en else "交接到研究项目",
        "draft": "Pass evidence gates before drafting" if is_en else "通过证据关口后再起草",
    }
    cohort_label = (
        "confirmed" if step_done["cohort"] else "waiting for cohort choice"
    ) if is_en else (
        "已确认" if step_done["cohort"] else "等待选择队列"
    )
    concepts_label = (
        f"{selected_count or _len_or_zero(study.get('modules'))} selected"
        if is_en else
        f"已选 {selected_count or _len_or_zero(study.get('modules'))} 个"
    )
    review_label = (
        f"{loaded_count} concepts · {patient_count} patients"
        if is_en else
        f"{loaded_count} 个概念 · {patient_count} 位患者"
    )
    agent_label = (
        "draft gated" if active_step == "draft" else "ready for handoff"
    ) if is_en else (
        "草稿已闸门锁定" if active_step == "draft" else "可交接"
    )
    return StudyWorkspaceSnapshot(
        active_step=active_step,
        active_step_label=labels[active_step],
        current_decision=decision_by_step[active_step],
        data_label=data_label,
        cohort_label=cohort_label,
        concepts_label=concepts_label,
        review_label=review_label,
        agent_label=agent_label,
        selected_concepts=selected_count,
        loaded_concepts=loaded_count,
        patient_count=patient_count,
        path_validated=path_validated,
        step_done=step_done,
    )


def _cohort_label_from_state(state: Mapping[str, Any], *, lang: str) -> str:
    is_en = lang == "en"
    inbound = state.get("research_agent_inbound_cohort")
    try:
        inbound_rows = len(inbound)  # type: ignore[arg-type]
    except Exception:
        inbound_rows = 0
    if inbound_rows:
        return f"{inbound_rows:,} ICU stays" if is_en else f"{inbound_rows:,} 个 ICU stay"
    built = state.get("research_agent_module_built")
    if isinstance(built, Mapping):
        df = built.get("df")
        try:
            built_rows = len(df)  # type: ignore[arg-type]
        except Exception:
            built_rows = 0
        if built_rows:
            return f"{built_rows:,} ICU stays" if is_en else f"{built_rows:,} 个 ICU stay"
    source = str(state.get("research_agent_cohort_source") or "").strip()
    if source:
        return source
    if _len_or_zero(state.get("loaded_concepts")):
        return "review workspace loaded" if is_en else "已加载审阅工作区"
    return "not selected" if is_en else "未选择"


def build_agent_project_snapshot(
    state: Mapping[str, Any],
    *,
    history_runs: Sequence[Mapping[str, Any]] | None = None,
    lang: str = "en",
) -> AgentProjectSnapshot:
    """Build the Agent Projects snapshot from session state and run summaries."""
    is_en = lang == "en"
    history_runs = history_runs or ()
    workbench = state.get("_agent_workbench")
    workbench = workbench if isinstance(workbench, Mapping) else {}
    audit = workbench.get("audit") if isinstance(workbench.get("audit"), Mapping) else {}
    counts = audit.get("counts") if isinstance(audit.get("counts"), Mapping) else {}
    gates = [gate for gate in audit.get("gates", []) or [] if isinstance(gate, Mapping)]
    review = audit.get("review_decision") if isinstance(audit.get("review_decision"), Mapping) else {}

    run_id = str(workbench.get("run_id") or state.get("research_agent_resume_run_id") or "").strip()
    run_dir = str(workbench.get("run_dir") or state.get("_agent_workbench_source_run_dir") or state.get("research_agent_resume_run_dir") or "").strip()
    if not run_id and run_dir:
        run_id = Path(run_dir).name
    question = str(
        workbench.get("research_question")
        or workbench.get("question")
        or state.get("research_agent_question")
        or ""
    ).strip()
    steps = [step for step in workbench.get("steps", []) or [] if isinstance(step, Mapping)]
    step_total = len(steps)
    step_ok = sum(1 for step in steps if str(step.get("status") or "").lower() == "ok")
    step_failed = sum(
        1
        for step in steps
        if str(step.get("status") or "").lower() in {"fail", "failed", "error", "blocked"}
    )
    evidence_count = _as_int(workbench.get("evidence_total"), _len_or_zero(workbench.get("evidence")))
    artifact_counts = workbench.get("artifact_counts") if isinstance(workbench.get("artifact_counts"), Mapping) else {}
    finding_errors = _as_int(counts.get("errors"))
    finding_warnings = _as_int(counts.get("warnings"))
    gates_blocked = sum(1 for gate in gates if gate.get("ok") is False)

    if run_id and (finding_errors or gates_blocked or step_failed):
        status = "Needs review" if is_en else "需要复核"
        status_tone = "warn"
    elif run_id and finding_warnings:
        status = "Review warnings" if is_en else "复核警告"
        status_tone = "warn"
    elif run_id:
        status = "Run ready" if is_en else "运行就绪"
        status_tone = "ready"
    elif question:
        status = "Setup in progress" if is_en else "配置中"
        status_tone = "warn"
    else:
        status = "New project" if is_en else "新项目"
        status_tone = "neutral"

    history_rows: list[dict[str, str]] = []
    for row in history_runs[:6]:
        run_id_value = str(row.get("run_id") or "").strip()
        if not run_id_value:
            continue
        history_rows.append(
            {
                "run_id": run_id_value,
                "status": str(row.get("status") or ""),
                "question": str(row.get("question") or ""),
                "evidence_count": str(row.get("evidence_count") or "0"),
                "run_dir": str(row.get("run_dir") or ""),
            }
        )

    project_title = question or ("Untitled Agent project" if is_en else "未命名研究项目")
    if len(project_title) > 96:
        project_title = project_title[:93] + "..."
    return AgentProjectSnapshot(
        project_id=run_id or "new_project",
        project_title=project_title,
        active_view=str(state.get("_ra_view") or "workbench"),
        status=status,
        status_tone=status_tone,
        run_id=run_id,
        run_dir=run_dir,
        question=question,
        cohort_label=_cohort_label_from_state(state, lang=lang),
        step_total=step_total,
        step_ok=step_ok,
        step_failed=step_failed,
        evidence_count=evidence_count,
        figure_count=_as_int(artifact_counts.get("figures")),
        table_count=_as_int(artifact_counts.get("tables")),
        finding_errors=finding_errors,
        finding_warnings=finding_warnings,
        gates_total=len(gates),
        gates_blocked=gates_blocked,
        review_decision=str(review.get("decision") or review.get("status") or ""),
        history_count=len(history_runs),
        history_runs=tuple(history_rows),
    )
