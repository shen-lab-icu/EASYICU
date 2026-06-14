"""Research Copilot — pure presentation helpers (labels, intros, stage/step rendering).

Extracted from `llm_chat.py` (Phase-6 split, 5th batch). Display-text and HTML
builders: option/step/database/disease/cohort labels, the step-by-step intro,
stage status/detail/card HTML, the rail step items, navigation completeness
predicates, workflow-snapshot HTML, and relative-time formatting. Every function
takes `study` / `state` / `snapshot` mappings as arguments and returns strings or
plain dict/list structures — **none reads the Streamlit `st` handle** — so the
test suite's `monkeypatch.setattr(llm_chat, "st", ...)` contract does not apply.

Cross-submodule leaf helpers (`_copilot_full_disease_options` from routing,
`_copilot_is_legacy_default_question` from sessions, `_copilot_normalize_database`
from intents) are imported directly — those modules never import this one, so no
cycle. The `COPILOT_*` constants and the one `_copilot_frame_question` helper that
still live in `llm_chat.py` are imported lazily inside the using functions
(routing.py pattern), so this module never imports `llm_chat` at load time.
`llm_chat.py` re-imports every name below, so all call sites keep working.
"""
from __future__ import annotations

import html
import re
from collections.abc import Mapping, MutableMapping
from datetime import datetime

from easyicu.webapp import copilot_engine as _copilot_engine
from easyicu.webapp.copilot.intents import _copilot_normalize_database
from easyicu.webapp.copilot.routing import _copilot_full_disease_options
from easyicu.webapp.copilot.sessions import _copilot_is_legacy_default_question


def _strip_module_label_icon(label: object) -> str:
    """Return the text label used in classic Step 3 without decorative icons."""
    text = re.sub(r"^[^\w\u4e00-\u9fff]+", "", str(label or "")).strip()
    return re.sub(r"\s+", " ", text)


def _copilot_step_by_step_intro(branch: str, lang: str) -> str:
    """Ask for the first human choice instead of preconfiguring the study."""
    if lang == "en":
        if branch == "crossdb":
            return (
                "Got it. I will keep this step-by-step and avoid choosing for you.\n\n"
                "**Step 1 · Research question:** what cohort and outcome signal do you want to compare across databases? "
                "You can name a disease group, a bedside score, a treatment exposure, or just describe the clinical question."
            )
        if branch == "quality":
            return (
                "Got it. I will keep this step-by-step and avoid choosing for you.\n\n"
                "**Step 1 · Audit target:** which data source, cohort, or concept family do you want to audit first? "
                "After that we will choose cohort scope, modules, extraction, and review in order."
            )
        return (
            "Got it. I will keep this step-by-step and avoid choosing for you.\n\n"
            "**Step 1 · Research question:** first choose the kind of study you want to build: outcome model, "
            "treatment exposure, cross-database comparison, data-quality audit, or your own question. After that I will "
            "ask for the endpoint/exposure if needed, then data source, cohort, feature modules, extraction, review, "
            "analysis, and draft gate in order."
        )
    if branch == "crossdb":
        return (
            "收到。我会按步骤来，不替你提前做选择。\n\n"
            "**第 1 步 · 研究问题：** 你想跨库比较哪个队列和结局信号？可以说疾病组、床旁评分、治疗暴露，"
            "也可以直接描述临床问题。"
        )
    if branch == "quality":
        return (
            "收到。我会按步骤来，不替你提前做选择。\n\n"
            "**第 1 步 · 审计目标：** 你想先审计哪个数据源、队列或概念家族？之后我们再依次选择队列范围、模块、提取和审阅。"
        )
    return (
        "收到。我会按步骤来，不替你提前做选择。\n\n"
        "**第 1 步 · 研究问题：** 先选择你要搭建哪类研究：结局建模、治疗暴露、跨库比较、数据质量审计，"
        "或直接描述自己的问题。之后我会按需继续问 endpoint/暴露，再依次选择数据源、队列、特征模块、提取、审阅、分析和草稿闸门。"
    )


def _copilot_database_label(database: object, lang: str) -> str:
    from easyicu.webapp.llm_chat import COPILOT_DATABASE_LABELS  # lazy
    key = _copilot_normalize_database(database)
    label_en, label_zh = COPILOT_DATABASE_LABELS.get(key, (key.upper(), key.upper()))
    return label_en if lang == "en" else label_zh


def _copilot_disease_label(disease: object, lang: str) -> str:
    from easyicu.webapp.llm_chat import COPILOT_DISEASE_OPTIONS  # lazy
    key = str(disease or "none").strip()
    if key in COPILOT_DISEASE_OPTIONS:
        label_en, label_zh = COPILOT_DISEASE_OPTIONS[key]
        return label_en if lang == "en" else label_zh
    # Fall back to the full classic template labels (ARDS, pneumonia, etc.).
    return _copilot_full_disease_options(lang).get(key, key)


def _copilot_data_source_choice_label(choice: str, lang: str) -> str:
    labels = {
        "prepared_path": ("Prepared data path", "prepared 数据路径"),
        "module_export": ("Module export folder", "模块导出文件夹"),
        "raw_files": ("Raw ICU files", "ICU 原始文件"),
    }
    label_en, label_zh = labels.get(choice, labels["prepared_path"])
    return label_en if lang == "en" else label_zh


def _copilot_cohort_is_empty(study: MutableMapping[str, object]) -> bool:
    return str(study.get("cohort_phase") or "ready") == "empty"


def _copilot_uses_eligible_cohort(study: Mapping[str, object]) -> bool:
    return str(study.get("cohort_strategy") or "").strip().lower() in {"eligible", "full", "all_eligible"}


def _copilot_cohort_label(study: Mapping[str, object], lang: str) -> str:
    if _copilot_uses_eligible_cohort(study):
        return "eligible real-data cohort" if lang == "en" else "真实数据合格队列"
    return (
        f"{int(study.get('patient_n') or 10)} stays"
        if lang == "en" else
        f"{int(study.get('patient_n') or 10)} 例 stay"
    )


def _normalized_copilot_workflow_snapshot(snapshot: Mapping[str, object], lang: str) -> dict[str, object]:
    from easyicu.webapp.llm_chat import _copilot_frame_question  # lazy
    clean: dict[str, object] = dict(snapshot)
    is_en = lang == "en"
    branch = str(clean.get("branch") or "")
    legacy_predict_label = "Predict " + "sepsis mortality"
    if branch == legacy_predict_label or "sepsis mortality" in branch.lower():
        clean["branch"] = "Model ICU outcomes" if is_en else "建模 ICU 结局"
    question = str(clean.get("question") or "")
    if _copilot_is_legacy_default_question(question):
        clean["question"] = _copilot_frame_question(
            {
                "branch": "predict",
                "window": "first 24h" if is_en else "前 24 小时",
                "outcome": "a prespecified ICU outcome" if is_en else "预设 ICU 结局",
            },
            lang,
        )
    if str(clean.get("step_title") or "") == "Cohort · all eligible stays":
        clean["step_title"] = "Cohort · needs confirmation"
        clean["step_detail"] = (
            "This old auto-selected scope is paused. Confirm the data source and cohort in chat before continuing."
        )
    if str(clean.get("step_title") or "") == "队列 · 全量合格 stay":
        clean["step_title"] = "队列 · 需要确认"
        clean["step_detail"] = "这个旧的自动队列范围已暂停。继续前请先在聊天中确认数据源和队列。"
    facts = clean.get("facts")
    if isinstance(facts, list):
        normalized_facts: list[object] = []
        for fact in facts:
            if isinstance(fact, Mapping):
                item = dict(fact)
                value = str(item.get("value") or "")
                if value == "all eligible stays":
                    item["value"] = "needs confirmation"
                elif value == "全量合格 stay":
                    item["value"] = "需要确认"
                normalized_facts.append(item)
            else:
                normalized_facts.append(fact)
        clean["facts"] = normalized_facts
    return clean


def _copilot_workflow_snapshot_html(snapshot: Mapping[str, object], lang: str) -> str:
    snapshot = _normalized_copilot_workflow_snapshot(snapshot, lang)
    # Slimmed card: the right-hand Study panel already carries the question,
    # full step map, evidence gate, and API state. The in-thread card keeps
    # ONLY the current step's "what this does" line so each step has a single,
    # non-redundant place to act (the edit controls render right below it).
    step_title = html.escape(str(snapshot.get("step_title") or ""))
    step_detail = html.escape(str(snapshot.get("step_detail") or ""))
    if not step_title and not step_detail:
        return ""
    eyebrow = html.escape("This step" if lang == "en" else "当前这一步")
    return (
        '<div class="eu-copilot-flow-card">'
        '<div class="flow-current">'
        f'<span class="flow-eyebrow">{eyebrow}</span>'
        f'<strong>{step_title}</strong>'
        f'<p>{step_detail}</p>'
        '</div>'
        '</div>'
    )


def _copilot_workflow_button_label(item: Mapping[str, object], lang: str) -> str:
    """Return a compact label for a workflow button without fake bullets."""
    label = str(item.get("label") or "")
    status = str(item.get("status") or "")
    if status == "done":
        return f"✓ {label}"
    if status == "active":
        return f"{label} · active" if lang == "en" else f"{label} · 当前"
    return label


def _copilot_visible_workflow_step_items(
    snapshot: Mapping[str, object],
    lang: str,
) -> list[dict[str, object]]:
    """Return only nearby workflow steps for the central chat card.

    The right rail already carries the complete study map. Keeping the central
    card to the previous/current/next step preserves editability without making
    the user evaluate the whole workflow at once.
    """
    normalized = _normalized_copilot_workflow_snapshot(snapshot, lang)
    steps = normalized.get("steps") if isinstance(normalized.get("steps"), list) else []
    step_items = [
        dict(step)
        for step in steps
        if isinstance(step, Mapping) and str(step.get("id") or "")
    ]
    if not step_items:
        return []
    active_step = str(normalized.get("active_step") or "")
    active_idx = next(
        (idx for idx, step in enumerate(step_items) if str(step.get("status") or "") == "active"),
        None,
    )
    if active_idx is None and active_step:
        active_idx = next(
            (idx for idx, step in enumerate(step_items) if str(step.get("id") or "") == active_step),
            None,
        )
    if active_idx is None:
        active_idx = 0
    visible_indices = [
        idx
        for idx in (active_idx - 1, active_idx, active_idx + 1)
        if 0 <= idx < len(step_items)
    ]
    return [step_items[idx] for idx in visible_indices]


def _copilot_concept_label_list(study: Mapping[str, object], *, limit: int = 5) -> list[str]:
    from easyicu.webapp.llm_chat import COPILOT_CONCEPT_LABELS  # lazy
    concepts = [
        str(item)
        for item in list(study.get("selected_concepts") or [])
        if str(item).strip()
    ]
    labels = [COPILOT_CONCEPT_LABELS.get(concept, concept) for concept in concepts]
    return labels[:limit]


def _copilot_action_material_icon(action: Mapping[str, object]) -> str:
    """Infer a Material icon for an option row from its id/label (polish gd-opt)."""
    from easyicu.webapp.llm_chat import _COPILOT_ACTION_ICON_RULES  # lazy
    text = (str(action.get("id") or "") + " " + str(action.get("label") or "")).lower()
    for keys, name in _COPILOT_ACTION_ICON_RULES:
        if any(k in text for k in keys):
            return f":material/{name}:"
    return ":material/arrow_forward:"


def _research_agent_source_label(source: str, lang: str) -> str:
    """Return the exact cohort-source radio label used by Research Agent."""
    is_en = lang == "en"
    if source == "handoff":
        return (
            "Use cohort prepared elsewhere in this session"
            if is_en else
            "使用本会话其他页面准备好的队列"
        )
    if source == "module":
        return (
            "Pick an EasyICU module export folder"
            if is_en else
            "选择 EasyICU 模块导出文件夹"
        )
    if source == "no_data":
        return (
            "I haven't extracted data yet — help me do it"
            if is_en else
            "我还没有提取数据，请帮我准备"
        )
    raise ValueError(f"Unknown Research Agent source label: {source}")


def _workflow_status_step(label: str, detail: str, status: str) -> str:
    safe_status = status if status in {"done", "active", "todo"} else "todo"
    return (
        f'<div class="eu-copilot-step {safe_status}">'
        '<span class="node"></span>'
        '<div>'
        f'<b>{html.escape(label)}</b>'
        f'<p>{html.escape(detail)}</p>'
        '</div>'
        '</div>'
    )


def _copilot_step_label(step: str, lang: str) -> str:
    from easyicu.webapp.llm_chat import COPILOT_STUDY_STEPS  # lazy
    label = dict(COPILOT_STUDY_STEPS).get(step, step)
    if lang == "en":
        return label
    return {
        "question": "研究问题",
        "data": "数据源",
        "cohort": "队列",
        "concepts": "特征模块",
        "extract": "提取",
        "review": "审阅",
        "analysis": "分析运行",
        "draft": "草稿闸门",
    }.get(step, label)


def _copilot_stage_status(study: MutableMapping[str, object], step: str) -> str:
    from easyicu.webapp.llm_chat import COPILOT_STEP_INDEX  # lazy
    active_step = str(study.get("step") or "question")
    active_idx = COPILOT_STEP_INDEX.get(active_step, 0)
    step_idx = COPILOT_STEP_INDEX.get(step, 0)
    if step_idx < active_idx or (step == "draft" and study.get("draft_signed")):
        return "done"
    if step_idx == active_idx:
        return "active"
    if step == "draft":
        return "locked"
    return "todo"


def _copilot_step_complete_for_navigation(
    study: Mapping[str, object],
    step: str,
) -> bool:
    """Return whether a rail step has enough real state to unlock later steps."""
    if step == "question":
        return bool(str(study.get("question") or "").strip())
    if step == "data":
        data_status = str(study.get("data_source_status") or "").strip()
        return data_status in {
            "pending_validation",
            "module_export_recorded",
            "conversion_needed",
        }
    if step == "cohort":
        return bool(study.get("cohort_configured")) or _copilot_uses_eligible_cohort(study)
    if step == "concepts":
        return bool(study.get("concepts_configured")) or bool(study.get("selected_concepts"))
    if step == "extract":
        return bool(study.get("extraction_configured"))
    if step == "review":
        return bool(study.get("review_configured"))
    if step == "analysis":
        return bool(study.get("analysis_configured"))
    if step == "draft":
        return bool(study.get("draft_signed"))
    return False


def _copilot_step_unlocked_for_navigation(
    study: Mapping[str, object],
    step: str,
) -> bool:
    """Allow jumps only to completed/current steps or the next satisfied step."""
    from easyicu.webapp.llm_chat import COPILOT_STEP_INDEX, COPILOT_STUDY_STEPS  # lazy
    if step == "question":
        return True
    # Depth gate: steps past the study's depth goal are locked ("beyond" in the
    # prototype) until the user takes the study further.
    if _copilot_engine.is_step_beyond_goal(study.get("depth"), step):
        return False
    active_step = str(study.get("step") or "question")
    if step == active_step:
        return True
    step_idx = COPILOT_STEP_INDEX.get(step, 0)
    sequence = [item[0] for item in COPILOT_STUDY_STEPS]
    prior_steps = sequence[:step_idx]
    return all(_copilot_step_complete_for_navigation(study, prior) for prior in prior_steps)


def _copilot_rail_step_items(
    study: Mapping[str, object],
    lang: str,
) -> list[dict[str, object]]:
    """Build the right-rail step model used for gated in-chat navigation."""
    from easyicu.webapp.llm_chat import COPILOT_STUDY_STEPS  # lazy
    items: list[dict[str, object]] = []
    is_en = lang == "en"
    for step, _label in COPILOT_STUDY_STEPS:
        status = _copilot_stage_status(dict(study), step)
        if (
            status == "todo"
            and step != "draft"
            and not _copilot_step_unlocked_for_navigation(study, step)
        ):
            status = "locked"
        label = _copilot_step_label(step, lang)
        if step == "draft":
            label = "Manuscript draft" if is_en else "手稿草稿"
        detail = ""
        if status in {"done", "active"}:
            if step == "question" and not str(study.get("question") or "").strip():
                detail = "Choose an outcome or describe a study" if is_en else "选择结局，或直接描述研究"
            elif step == "cohort" and not bool(study.get("cohort_configured")):
                detail = "Waiting for cohort choices" if is_en else "等待队列选择"
            elif step == "concepts" and not bool(study.get("concepts_configured")):
                detail = "Waiting for module choices" if is_en else "等待模块选择"
            else:
                detail = _copilot_stage_detail(dict(study), step, lang)
        elif status == "locked":
            detail = (
                "Complete the previous step first"
                if is_en else
                "先完成上一步"
            )
        items.append({
            "id": step,
            "label": label,
            "detail": detail,
            "status": status,
            "is_goal": step == _copilot_engine.copilot_goal_step(study.get("depth")),
            "beyond_goal": _copilot_engine.is_step_beyond_goal(study.get("depth"), step),
            "unlocked": _copilot_step_unlocked_for_navigation(study, step),
            "icon": {
                "question": "spark",
                "data": "flask",
                "cohort": "hexagon",
                "concepts": "layers",
                "extract": "stack",
                "review": "eye",
                "analysis": "robot",
                "draft": "shield",
            }.get(step, "grid"),
        })
    return items


def _copilot_stage_detail(study: MutableMapping[str, object], step: str, lang: str) -> str:
    from easyicu.webapp.llm_chat import COPILOT_BRANCH_CONFIG, COPILOT_DEFAULT_MODULES  # lazy
    is_en = lang == "en"
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])
    if step == "question":
        return str(study.get("question") or (config["chip"] if is_en else config["question_zh"]))
    if step == "data":
        mode = str(study.get("data_mode") or "real")
        if mode == "real":
            choice = str(study.get("data_source_choice") or "")
            status = str(study.get("data_source_status") or "")
            db_label = _copilot_database_label(study.get("database") or "miiv", lang)
            if status == "awaiting_path":
                if choice == "module_export":
                    return "module export field open" if is_en else "模块导出输入框已打开"
                if choice == "raw_files":
                    return "raw files field open" if is_en else "原始文件输入框已打开"
                return "prepared path field open" if is_en else "prepared 路径输入框已打开"
            if status == "module_export_recorded":
                return f"{db_label} · module export recorded" if is_en else f"{db_label} · 模块导出已记录"
            if status == "conversion_needed":
                return f"{db_label} · raw files recorded · conversion needed" if is_en else f"{db_label} · 原始文件已记录 · 需要转换"
            if status == "pending_validation":
                return f"{db_label} · prepared path recorded · pending validation" if is_en else f"{db_label} · prepared 路径已记录 · 待验证"
            return "choose source type" if is_en else "选择来源类型"
        return "real data source not connected" if is_en else "真实数据源尚未连接"
    if step == "cohort":
        if _copilot_cohort_is_empty(study):
            return "0 stays · strict filters" if is_en else "0 例 stay · 严格过滤"
        if branch == "crossdb":
            return f"{int(study.get('db_count') or 6)} databases · shared cohort definition" if is_en else f"{int(study.get('db_count') or 6)} 个数据库 · 共享队列定义"
        if _copilot_uses_eligible_cohort(study):
            return "eligible real-data cohort · first ICU stay · first 24h" if is_en else "真实数据合格队列 · 首次 ICU · 前 24h"
        return f"{int(study.get('patient_n') or 10)} stays · first ICU stay · first 24h" if is_en else f"{int(study.get('patient_n') or 10)} 例 stay · 首次 ICU · 前 24h"
    if step == "concepts":
        modules = list(study.get("modules") or COPILOT_DEFAULT_MODULES)
        return f"{len(modules)} modules · coverage audited before analysis" if is_en else f"{len(modules)} 个模块 · 分析前审计覆盖率"
    if step == "extract":
        if _copilot_uses_eligible_cohort(study):
            return "eligible first-24h frames · frozen normalized export" if is_en else "合格队列前 24h 数据帧 · 冻结标准化导出"
        n = int(study.get("patient_n") or 10)
        return f"{n * 24} time points · frozen normalized frames" if is_en else f"{n * 24} 个时间点 · 冻结标准化数据帧"
    if step == "review":
        if branch == "crossdb":
            return "availability matrix and distribution deltas" if is_en else "可用性矩阵与分布差异"
        if branch == "quality":
            return "coverage, ranges, missingness, density" if is_en else "覆盖率、范围、缺失和时间密度"
        return "Table 1, time series, patient overview, quality flags" if is_en else "Table 1、时间序列、患者概览和质量标记"
    if step == "analysis":
        return "5 deterministic steps · evidence contract" if is_en else "5 个确定性步骤 · 证据契约"
    if step == "draft":
        if study.get("draft_signed"):
            return "unlocked after local sign-off" if is_en else "本地确认后已解锁"
        return "locked until checks pass and a human signs off" if is_en else "证据检查和人工确认前保持锁定"
    return ""


def _copilot_stage_card_html(
    study: MutableMapping[str, object],
    step: str,
    lang: str,
    *,
    compact: bool = False,
) -> str:
    from easyicu.webapp.llm_chat import COPILOT_BRANCH_CONFIG, COPILOT_STRICT_COHORT_FILTERS  # lazy
    is_en = lang == "en"
    status = _copilot_stage_status(study, step)
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])
    why = str(config["why"].get(step, ""))
    status_label = {
        "done": "done" if is_en else "完成",
        "active": "active" if is_en else "当前",
        "locked": "locked" if is_en else "锁定",
        "todo": "queued" if is_en else "待定",
    }[status]
    if compact:
        return (
            f'<div class="eu-copilot-stage collapsed {status}">'
            '<span class="mark"></span>'
            f'<b>{html.escape(_copilot_step_label(step, lang))}</b>'
            f'<em>{html.escape(_copilot_stage_detail(study, step, lang))}</em>'
            '</div>'
        )
    if step == "cohort" and _copilot_cohort_is_empty(study):
        filters = list(study.get("cohort_filters") or COPILOT_STRICT_COHORT_FILTERS)
        chips = "".join(f'<span class="eu-state-chip solid">{html.escape(str(item))}</span>' for item in filters)
        title = "No patients match those filters" if is_en else "没有患者匹配这些过滤条件"
        detail = (
            '"Sepsis-3 + age >= 80" is empty in this demo set/export. Loosen a constraint and I will re-match.'
            if is_en else
            "“Sepsis-3 + 年龄 ≥ 80” 在这个演示数据/导出中为空。放宽一个条件后我会重新匹配。"
        )
        return (
            f'<div class="eu-copilot-stage active" data-step="{html.escape(step)}">'
            '<div class="stage-head">'
            '<span class="mark"></span>'
            '<div>'
            f'<b>{html.escape(_copilot_step_label(step, lang))}</b>'
            f'<p>{html.escape(_copilot_stage_detail(study, step, lang))}</p>'
            '</div>'
            f'<span class="stage-status">{html.escape(status_label)}</span>'
            '</div>'
            '<div class="eu-state-hero nodata eu-copilot-nodata">'
            '<div class="glyph"></div>'
            f'<div class="st-t">{html.escape(title)}</div>'
            f'<div class="st-d">{html.escape(detail)}</div>'
            f'<div class="filter-recap">{chips}</div>'
            '</div>'
            f'<div class="stage-why"><span>{"WHY THIS STEP" if is_en else "为什么做这一步"}</span>{html.escape(why)}</div>'
            '</div>'
        )
    return (
        f'<div class="eu-copilot-stage {status}" data-step="{html.escape(step)}">'
        '<div class="stage-head">'
        '<span class="mark"></span>'
        '<div>'
        f'<b>{html.escape(_copilot_step_label(step, lang))}</b>'
        f'<p>{html.escape(_copilot_stage_detail(study, step, lang))}</p>'
        '</div>'
        f'<span class="stage-status">{html.escape(status_label)}</span>'
        '</div>'
        f'<div class="stage-why"><span>{"WHY THIS STEP" if is_en else "为什么做这一步"}</span>{html.escape(why)}</div>'
        '</div>'
    )


def _copilot_relative_time(iso_str: str, lang: str) -> str:
    """Human relative time ('2h ago' / '2小时前') from an ISO timestamp."""
    raw = str(iso_str or "").strip()
    if not raw:
        return ""
    is_en = lang == "en"
    try:
        ts = datetime.fromisoformat(raw)
    except ValueError:
        return raw[:16].replace("T", " ")
    delta = datetime.now() - ts
    secs = max(0, int(delta.total_seconds()))
    mins, hours, days = secs // 60, secs // 3600, secs // 86400
    if secs < 60:
        return "just now" if is_en else "刚刚"
    if mins < 60:
        return f"{mins}m ago" if is_en else f"{mins}分钟前"
    if hours < 24:
        return f"{hours}h ago" if is_en else f"{hours}小时前"
    if days == 1:
        return "yesterday" if is_en else "昨天"
    if days < 7:
        return f"{days}d ago" if is_en else f"{days}天前"
    return ts.strftime("%b %d") if is_en else ts.strftime("%m-%d")
