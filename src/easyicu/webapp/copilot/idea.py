"""Research Copilot idea-recommendation / UI-action builders (batch-10 split from llm_chat.py).

Pure builders for the literature-seeded idea cluster and the generic
``_suggest_ui_actions`` action factory. **Zero ``st`` reference** (AST-verified),
so they live outside ``llm_chat`` without breaking the
``monkeypatch.setattr(llm_chat, "st", fake)`` test contract. ``llm_chat``
re-imports every name so call sites and tests keep using ``llm_chat.<fn>``.

Module-level constants (COPILOT_CONCEPT_LABELS / COPILOT_IDEA_CANDIDATES /
ALL_PRESET_GROUP_KEYS / SEPSIS_PRESET_CONCEPTS) stay in ``llm_chat``; the
consumers import them lazily in-body to avoid an import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping  # noqa: F401  (annotations)
from datetime import datetime  # noqa: F401

from easyicu.webapp.copilot.actions import _copilot_guided_choice_actions
from easyicu.webapp.copilot.handoff import _copilot_frame_question
from easyicu.webapp.copilot.intents import (
    _copilot_idea_topic,
    _copilot_modules_for_concepts,
    _copilot_parse_idea_selection,
    _copilot_usage_help_requested,
    _infer_db_from_text,
)
from easyicu.webapp.copilot.sessions import _ensure_copilot_study_state

def _copilot_reply(
    study: MutableMapping[str, object],
    body: str,
    lang: str,
    *,
    include_status: bool = True,
) -> str:
    question = str(study.get("question") or _copilot_frame_question(study, lang)).strip()
    if question:
        study["question"] = question
    if not include_status:
        return body
    return body


def _is_idea_exploration_request(text: str) -> bool:
    """Return True when a short prompt asks for the Agent idea-mining path."""
    raw = text or ""
    text_l = raw.lower()
    english_terms = (
        "idea exploration",
        "idea mining",
        "literature-derived",
        "review-derived",
        "review excerpt",
        "editorial excerpt",
        "discovery triage",
        "candidate hypothesis",
        "hypothesis candidate",
        "prior art",
        "preregistration registry",
        "registry gate",
    )
    chinese_terms = (
        "idea 探索",
        "idea探索",
        "idea 挖掘",
        "idea挖掘",
        "文献发现",
        "文献探索",
        "综述摘录",
        "综述 idea",
        "综述idea",
        "editorial",
        "候选假设",
        "查新",
        "创新点",
        "研究想法",
        "研究idea",
        "discovery",
    )
    if any(term in text_l for term in english_terms):
        return True
    if any(term in raw for term in chinese_terms):
        return True
    return "idea" in text_l and any(
        term in text_l
        for term in ("explore", "exploration", "mining", "triage", "discovery", "hypothesis")
    )


def _copilot_idea_candidates_for_prompt(text: str) -> list[dict[str, object]]:
    from easyicu.webapp.llm_chat import COPILOT_IDEA_CANDIDATES  # lazy: avoid import cycle
    topic = _copilot_idea_topic(text)
    candidates = COPILOT_IDEA_CANDIDATES.get(topic) or COPILOT_IDEA_CANDIDATES["general"]
    return [dict(item) for item in candidates]


def _copilot_store_idea_context(
    state: MutableMapping[str, object],
    prompt: str,
    candidates: list[dict[str, object]],
) -> None:
    state["_copilot_idea_context"] = {
        "topic": _copilot_idea_topic(prompt),
        "prompt": prompt,
        "candidates": candidates,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _copilot_idea_recommendation_reply(
    prompt: str,
    lang: str,
    state: MutableMapping[str, object],
) -> tuple[str, list[dict[str, object]]]:
    """Keep idea discovery inside Copilot chat instead of routing to Research Agent."""
    from easyicu.webapp.llm_chat import COPILOT_CONCEPT_LABELS  # lazy: avoid import cycle
    candidates = _copilot_idea_candidates_for_prompt(prompt)
    _copilot_store_idea_context(state, prompt, candidates)
    state["_copilot_suppress_next_snapshot"] = True
    is_en = lang == "en"
    lines: list[str] = []
    if is_en:
        lines.append("I will keep this inside Copilot: no jump button, no preset decision yet.")
        lines.append("")
        lines.append("Here are three executable directions:")
        for idx, candidate in enumerate(candidates, start=1):
            concepts = ", ".join(
                COPILOT_CONCEPT_LABELS.get(str(item), str(item))
                for item in list(candidate.get("concepts") or [])
            )
            lines.append(
                f"{idx}. **{candidate.get('title_en')}** — {candidate.get('question_en')}\n"
                f"   Why: {candidate.get('why_en')}\n"
                f"   Candidate modules: {concepts}"
            )
        lines.append("")
        lines.append(
            "Reply with `pick 1`, `pick 2`, or `pick 3`, or edit one in plain language. "
            "After that I will frame the study, then ask for data source, cohort, and feature confirmation in chat."
        )
    else:
        lines.append("我会留在 Copilot 里完成：不甩给跳转按钮，也不先替你定题。")
        lines.append("")
        lines.append("先给你 3 个可以落到 EasyICU 工作流里的方向：")
        for idx, candidate in enumerate(candidates, start=1):
            concepts = "、".join(
                COPILOT_CONCEPT_LABELS.get(str(item), str(item))
                for item in list(candidate.get("concepts") or [])
            )
            lines.append(
                f"{idx}. **{candidate.get('title_zh')}**：{candidate.get('question_zh')}\n"
                f"   为什么适合：{candidate.get('why_zh')}\n"
                f"   候选模块：{concepts}"
            )
        lines.append("")
        lines.append(
            "你直接回复“选 1 / 选 2 / 选 3”，也可以说“把第 2 个改成 AKI/RRT 结局”。"
            "我会先把选题写入聊天状态，然后继续在聊天里问数据源、队列和变量，不跳转。"
        )
    return "\n\n".join(lines), []


def _copilot_handle_idea_selection(
    prompt: str,
    lang: str,
    state: MutableMapping[str, object],
) -> tuple[str, list[dict[str, object]]] | None:
    from easyicu.webapp.llm_chat import COPILOT_CONCEPT_LABELS  # lazy: avoid import cycle
    context = state.get("_copilot_idea_context")
    if not isinstance(context, Mapping):
        return None
    candidates = context.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    selected_idx = _copilot_parse_idea_selection(prompt)
    if selected_idx is None or selected_idx < 0 or selected_idx >= len(candidates):
        return None
    raw_candidate = candidates[selected_idx]
    if not isinstance(raw_candidate, Mapping):
        return None
    candidate = dict(raw_candidate)
    study = _ensure_copilot_study_state(state)
    study["branch"] = str(candidate.get("branch") or "predict")
    study["question"] = str(candidate.get("question_en" if lang == "en" else "question_zh") or "")
    study["step"] = "data"
    study["data_mode"] = "real"
    study["cohort_configured"] = False
    study["concepts_configured"] = False
    study["idea_candidate_id"] = str(candidate.get("id") or "")
    suggested_concepts = [
        str(item)
        for item in list(candidate.get("concepts") or [])
        if str(item).strip()
    ]
    study["suggested_concepts"] = suggested_concepts
    study["modules"] = _copilot_modules_for_concepts(suggested_concepts)
    study.pop("selected_concepts", None)
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    state["_copilot_guided_study"] = study
    state["_copilot_idea_context"] = {
        **dict(context),
        "selected_id": study["idea_candidate_id"],
        "selected_at": datetime.now().isoformat(timespec="seconds"),
    }
    concept_labels = [
        COPILOT_CONCEPT_LABELS.get(concept, concept)
        for concept in suggested_concepts
    ]
    if lang == "en":
        body = (
            f"Good. I framed option {selected_idx + 1} as:\n\n"
            f"**{study['question']}**\n\n"
            "I have not selected the data source, cohort denominator, or modules for you yet. "
            f"Candidate modules are: **{', '.join(concept_labels)}**.\n\n"
            "Next, choose the real-data source below or paste `set data path /...` in this chat. "
            "If the candidate modules look right, say `use these modules`."
        )
    else:
        body = (
            f"好，我先把第 {selected_idx + 1} 个方向整理成研究问题：\n\n"
            f"**{study['question']}**\n\n"
            "我还没有替你选择数据源、队列分母或锁定变量。"
            f"候选模块是：**{'、'.join(concept_labels)}**。\n\n"
            "下一步请在下方选择真实数据源，或直接粘贴 `set data path /...`。"
            "如果这些候选模块可以，就回复“用这些变量”。"
        )
    return _copilot_reply(study, body, lang, include_status=False), _copilot_guided_choice_actions(study, lang)


def _suggest_ui_actions(prompt: str, answer: str, lang: str) -> list[dict[str, object]]:
    """Suggest in-app navigation or preset actions."""
    from easyicu.webapp.llm_chat import (  # lazy: avoid import cycle
        ALL_PRESET_GROUP_KEYS,
        SEPSIS_PRESET_CONCEPTS,
    )
    prompt_l = (prompt or '').lower()
    answer_l = (answer or '').lower()
    combined = f"{prompt_l}\n{answer_l}"
    actions: list[dict[str, object]] = []

    def add_nav(action_id: str, label_en: str, label_zh: str):
        if any(item["id"] == action_id for item in actions):
            return
        actions.append({
            "id": action_id,
            "kind": "nav",
            "label": label_en if lang == "en" else label_zh,
        })

    def add_preset(
        action_id: str,
        label_en: str,
        label_zh: str,
        payload: dict[str, object],
        scroll_to: str = "tutorial",
    ):
        if any(item["id"] == action_id for item in actions):
            return
        actions.append({
            "id": action_id,
            "kind": "preset",
            "label": label_en if lang == "en" else label_zh,
            "payload": payload,
            "scroll_to": scroll_to,
        })

    def add_workflow(action_id: str, label_en: str, label_zh: str, workflow: str):
        if any(item["id"] == action_id for item in actions):
            return
        actions.append({
            "id": action_id,
            "kind": "workflow",
            "label": label_en if lang == "en" else label_zh,
            "workflow": workflow,
        })

    def add_agent_handoff(action_id: str, label_en: str, label_zh: str):
        if any(item["id"] == action_id for item in actions):
            return
        actions.append({
            "id": action_id,
            "kind": "agent_handoff",
            "label": label_en if lang == "en" else label_zh,
        })

    dictionary_requested = any(key in prompt_l for key in ["字典", "数据字典", "dictionary", "feature list", "concept list"])
    usage_help_requested = _copilot_usage_help_requested(prompt)
    tutorial_requested = (
        any(key in prompt_l for key in ["tutorial", "教程", "step", "步骤", "how do i", "怎么做", "workflow", "流程", "guide", "使用"])
        and not usage_help_requested
    )
    viz_requested = any(key in prompt_l for key in [
        "quick visualization", "快速可视化", "load data", "加载数据", "visualization",
        "visualize", "visualise", "plot", "可视化", "图表", "数据分析", "分析我的数据",
    ])
    cohort_requested = any(key in prompt_l for key in ["cohort", "队列", "compare", "comparison", "dashboard", "仪表板"])
    export_requested = any(key in prompt_l for key in ["export", "导出"])
    demo_requested = any(key in prompt_l for key in [
        "demo", "演示", "模拟", "try first", "load demo", "populate", "sample workspace",
        "样例", "示例", "先跑", "快速体验",
    ])
    extraction_requested = any(key in prompt_l for key in [
        "data extraction", "extract", "提取", "抽取", "data source", "数据源",
        "real data", "真实数据", "connect data", "连接数据",
    ])
    agent_requested = any(key in prompt_l for key in [
        "research agent", "agent", "智能体", "manuscript", "draft", "草稿",
        "evidence", "证据", "run study", "guided study", "do it for me",
        "copilot", "一键", "帮我跑", "自动跑",
    ])
    idea_requested = _is_idea_exploration_request(combined)
    guided_demo_requested = any(key in prompt_l for key in [
        "run the whole demo", "whole demo", "guided demo", "autopilot",
        "do it for me", "just do it", "跑完整演示", "完整演示", "帮我跑", "自动跑",
    ])

    if dictionary_requested or (
        any(key in answer_l for key in ["data dictionary", "数据字典", "concept dictionary"]) and
        any(key in prompt_l for key in ["where", "在哪", "在哪里", "怎么找", "how to find", "查看"])
    ):
        add_nav("home_dict", "Open Data Dictionary", "打开数据字典")

    if tutorial_requested and not dictionary_requested:
        add_nav("tutorial", "Open Tutorial", "打开教程")

    if viz_requested:
        add_nav("viz", "Open Quick Visualization", "前往快速可视化")

    if cohort_requested:
        add_nav("cohort", "Open Cohort Analysis", "前往队列分析")

    if export_requested:
        add_nav("tutorial", "Open Export Guide", "打开导出教程")

    if guided_demo_requested:
        add_workflow("workflow_guided_demo", "Run Guided Demo", "运行引导演示", "guided_demo")

    # Idea/recommendation prompts stay inside Copilot chat. Do not suggest a page jump here.

    if demo_requested and (viz_requested or "workspace" in prompt_l or "加载" in prompt_l or "populate" in prompt_l):
        add_workflow("workflow_demo_review", "Load Demo Review Workspace", "加载演示审阅工作区", "demo_review")
    elif demo_requested or (extraction_requested and any(key in prompt_l for key in ["start", "开始", "first", "入口"])):
        add_workflow("workflow_demo_extraction", "Start with Demo Extraction", "从演示提取开始", "demo_extraction")

    if extraction_requested and not demo_requested:
        add_workflow("workflow_real_extraction", "Open Real Data Extraction", "打开真实数据提取", "real_extraction")

    if agent_requested and not idea_requested:
        add_agent_handoff("agent_handoff", "Hand off to Research Agent", "交给 Research Agent")

    target_db = _infer_db_from_text(prompt_l)
    is_all_features_request = any(key in prompt_l for key in [
        "所有临床指标", "所有特征", "全部特征", "all clinical features",
        "all concepts", "all indicators", "all features",
    ])
    is_sepsis_extract_request = (
        any(key in prompt_l for key in ["sepsis", "脓毒症", "septic shock", "脓毒性休克"])
        and any(key in prompt_l for key in ["提取", "抽取", "export", "导出", "select", "选择"])
    )

    if target_db == "miiv" and is_all_features_request:
        add_preset(
            "preset_miiv_all",
            "Prepare MIMIC-IV Full Feature Selection",
            "预设 MIMIC-IV 全量特征选择",
            {
                "kind": "feature_preset",
                "database": "miiv",
                "group_keys": ALL_PRESET_GROUP_KEYS,
                "notice_en": "Switched toward Real Data with a MIMIC-IV full-feature preset ready. Next: choose Real Data mode if needed, fill the data path, run Validate Data Path or Convert & Setup, then finish Step 2. The feature preset will appear automatically in Step 3.",
                "notice_zh": "已切换到面向真实数据的 MIMIC-IV 全量特征预设。下一步：如果还没在真实数据模式，请先切换；然后填写数据路径，执行“验证数据路径”或“转换并设置”，完成步骤2后，步骤3会自动出现这套特征预设。",
                "apply_notice_en": "Your MIMIC-IV full-feature preset is now loaded in Step 3. Review the checked features, then confirm selection.",
                "apply_notice_zh": "MIMIC-IV 全量特征预设已加载到步骤3。请检查已勾选特征，然后确认选择。",
            },
        )

    if target_db == "miiv" and is_sepsis_extract_request:
        add_preset(
            "preset_miiv_sepsis",
            "Prepare MIMIC-IV Sepsis Feature Set",
            "预设 MIMIC-IV Sepsis 特征集",
            {
                "kind": "feature_preset",
                "database": "miiv",
                "group_keys": [
                    "sepsis3_sofa2",
                    "sepsis3_sofa1",
                    "sepsis_shared",
                    "sofa2_score",
                    "sofa1_score",
                    "vitals",
                    "respiratory",
                    "blood_gas",
                    "chemistry",
                    "hematology",
                    "vasopressors",
                    "medications",
                    "renal",
                    "outcome",
                ],
                "concepts": SEPSIS_PRESET_CONCEPTS,
                "notice_en": "Switched toward Real Data with a MIMIC-IV Sepsis preset ready. Next: choose Real Data mode if needed, fill the data path, run Validate Data Path or Convert & Setup, then finish Step 2. The Sepsis feature preset will appear automatically in Step 3.",
                "notice_zh": "已切换到面向真实数据的 MIMIC-IV Sepsis 预设。下一步：如果还没在真实数据模式，请先切换；然后填写数据路径，执行“验证数据路径”或“转换并设置”，完成步骤2后，步骤3会自动出现这套 Sepsis 特征预设。",
                "apply_notice_en": "Your MIMIC-IV Sepsis feature preset is now loaded in Step 3. Review the checked concepts, then confirm selection.",
                "apply_notice_zh": "MIMIC-IV Sepsis 特征预设已加载到步骤3。请检查已勾选概念，然后确认选择。",
            },
        )

    return actions[:3]
