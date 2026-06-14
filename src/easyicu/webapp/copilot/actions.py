"""Research Copilot guided-action / prompt builders (batch-9 split from llm_chat.py).

Pure builders: every function here takes ``study``/``state`` mappings and returns
action dicts / prompt lists. **Zero ``st`` reference** — they never touch
``st.session_state`` (verified by AST), so they are safe to live outside
``llm_chat`` without breaking the ``monkeypatch.setattr(llm_chat, "st", fake)``
test contract. ``llm_chat`` re-imports every name below so all call sites and
tests keep using ``llm_chat.<fn>`` unchanged.

The two ``COPILOT_FEATURE_MODULE_*`` constants stay in ``llm_chat`` (other
modules use them); the single consumer here imports them lazily in-body to
avoid an import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping  # noqa: F401  (used in annotations)

from easyicu.webapp.copilot.intents import _copilot_feature_pack_label
from easyicu.webapp.copilot.presentation import (
    _copilot_cohort_is_empty,
    _copilot_uses_eligible_cohort,
)

def _copilot_feature_module_action_keys() -> list[str]:
    """Small real-module shortcut set; the inline Step 3 form renders every module."""
    from easyicu.webapp.llm_chat import (  # lazy: avoid import cycle
        COPILOT_FEATURE_MODULE_ACTION_KEYS,
        COPILOT_FEATURE_MODULE_PACKS,
    )
    return [key for key in COPILOT_FEATURE_MODULE_ACTION_KEYS if key in COPILOT_FEATURE_MODULE_PACKS]


def _copilot_feature_module_prompt(pack_key: str, lang: str) -> str:
    label = _copilot_feature_pack_label(pack_key, lang)
    return f"Add feature module: {label}." if lang == "en" else f"选择特征模块：{label}。"


def _copilot_feature_module_actions(lang: str) -> list[dict[str, object]]:
    actions = [
        _copilot_prompt_action(
            f"choice_modules_{key}",
            _copilot_feature_pack_label(key, "en"),
            _copilot_feature_pack_label(key, "zh"),
            _copilot_feature_module_prompt(key, "en"),
            _copilot_feature_module_prompt(key, "zh"),
            lang,
        )
        for key in _copilot_feature_module_action_keys()
    ]
    actions.append(
        _copilot_prompt_action(
            "choice_modules_suggested",
            "Use model-suggested modules",
            "使用模型推荐模块",
            "use these modules",
            "用这些变量",
            lang,
        )
    )
    return actions


def _normalized_copilot_message_actions(
    actions: object,
    lang: str,
) -> list[dict[str, object]]:
    """Normalize stale stored Copilot action buttons before rendering them."""
    if not isinstance(actions, list):
        return []
    normalized: list[dict[str, object]] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        item = dict(action)
        label = str(item.get("label") or "")
        workflow = str(item.get("workflow") or "")
        action_id = str(item.get("id") or "")
        if workflow == "agent_idea_exploration" or action_id == "workflow_agent_idea":
            continue
        if item.get("kind") == "preset" and action_id == "preset_miiv_sepsis":
            continue
        if workflow == "study_strict_filters" or label.startswith("Restrict: Sepsis-3"):
            continue
        if (
            workflow == "real_extraction"
            and action_id == "workflow_real_extraction"
            and "Real Data Setup" in label
        ):
            item["id"] = "workflow_real_data_chat_setup"
            item["workflow"] = "real_data_chat_setup"
            item["label"] = "Set data path in chat" if lang == "en" else "在聊天中设置路径"
        normalized.append(item)
    return normalized


def _copilot_current_prompt_actions(
    state: Mapping[str, object],
    lang: str,
) -> list[dict[str, object]]:
    """Return the currently valid guided choices without mutating session state."""
    raw_study = state.get("_copilot_guided_study")
    study = raw_study if isinstance(raw_study, Mapping) else {}
    return _copilot_guided_choice_actions(study, lang)


def _copilot_message_actions_for_current_step(
    actions: object,
    lang: str,
    state: Mapping[str, object],
    *,
    is_latest: bool,
) -> list[dict[str, object]]:
    """Render only current-step choices inside Copilot assistant replies.

    Stored chat messages can outlive the workflow step they were created for. In
    the Copilot page, stale prompt buttons are more confusing than helpful, so
    old choices are filtered to the current guided step. If the newest assistant
    reply came from a generic/local answer without actions, inject the current
    step choices so the reply remains actionable in-place.
    """
    normalized = _normalized_copilot_message_actions(actions, lang)
    current_actions = _copilot_current_prompt_actions(state, lang)
    current_ids = {str(action.get("id") or "") for action in current_actions}
    recovery_ids = {"choice_retry_routing"}
    recovery_actions = [
        action
        for action in normalized
        if str(action.get("kind") or "") == "copilot_prompt"
        and str(action.get("id") or "") in recovery_ids
    ]
    step_actions = [
        action
        for action in normalized
        if str(action.get("kind") or "") == "copilot_prompt"
        and str(action.get("id") or "") in current_ids
    ]
    if step_actions:
        if is_latest and recovery_actions:
            return recovery_actions + [
                action
                for action in step_actions
                if str(action.get("id") or "") not in recovery_ids
            ]
        return step_actions
    if is_latest and recovery_actions:
        current_without_recovery = [
            action
            for action in current_actions
            if str(action.get("id") or "") not in recovery_ids
        ]
        return recovery_actions + current_without_recovery
    if is_latest and current_actions:
        return current_actions
    if not is_latest:
        return []
    return [
        action
        for action in normalized
        if str(action.get("kind") or "") not in {"workflow", "agent_handoff", "preset"}
    ]


def _copilot_study_actions(study: MutableMapping[str, object], lang: str) -> list[dict[str, object]]:
    is_en = lang == "en"
    step = str(study.get("step") or "question")
    actions: list[dict[str, object]] = []

    def workflow(action_id: str, label_en: str, label_zh: str, target: str) -> None:
        if any(item["id"] == action_id for item in actions):
            return
        actions.append({
            "id": action_id,
            "kind": "workflow",
            "label": label_en if is_en else label_zh,
            "workflow": target,
        })

    if step == "cohort" and _copilot_cohort_is_empty(study):
        workflow("workflow_study_loosen_filters", "Loosen filters", "放宽过滤条件", "study_loosen_filters")
        workflow("workflow_study_defaults", "Back to defaults", "恢复默认", "study_loosen_filters")
        return actions[:3]
    if step in {"data", "cohort", "concepts", "extract"}:
        workflow("workflow_study_extract", "Classic workspace", "经典工作区", "study_extract")
    if step == "draft" and study.get("draft_signed"):
        workflow("workflow_study_draft", "Open draft", "打开草稿", "study_draft")
    if step in {"review", "analysis", "draft"}:
        workflow("workflow_study_review", "Open Review Workspace", "打开审阅工作区", "study_review")
    if step == "draft" and not study.get("draft_signed"):
        workflow("workflow_study_signoff", "Review & sign off", "审阅并确认", "study_signoff")
    if step in {"concepts", "extract", "review", "analysis", "draft"}:
        actions.append({
            "id": "agent_handoff",
            "kind": "agent_handoff",
            "label": "Agent setup" if is_en else "Agent 配置",
        })
    if str(study.get("data_mode")) == "real":
        workflow("workflow_real_data_chat_setup", "Set data path in chat", "在聊天中设置路径", "real_data_chat_setup")
    return actions[:3]


def _copilot_prompt_action(
    action_id: str,
    label_en: str,
    label_zh: str,
    prompt_en: str,
    prompt_zh: str,
    lang: str,
    *,
    desc_en: str = "",
    desc_zh: str = "",
) -> dict[str, object]:
    """Build an action that continues the Copilot chat instead of navigating.

    Optional `desc_*` gives the option card a polish gd-opt style second line
    (one-line explanation). It is rendered only inside the button and never used
    as the displayed user message (the clean `label` is).
    """
    action: dict[str, object] = {
        "id": action_id,
        "kind": "copilot_prompt",
        "label": label_en if lang == "en" else label_zh,
        "prompt": prompt_en if lang == "en" else prompt_zh,
    }
    desc = desc_en if lang == "en" else desc_zh
    if desc:
        action["desc"] = desc
    return action


def _copilot_nonblocking_goal_fallback_actions(lang: str) -> list[dict[str, object]]:
    return [
        _copilot_prompt_action(
            "route_fallback_data_path",
            "Set real data path",
            "设置真实路径",
            "I have a prepared data path.",
            "我有 prepared 数据路径。",
            lang,
        ),
        _copilot_prompt_action(
            "route_fallback_cohort",
            "Choose cohort",
            "选择队列",
            "Walk me through the cohort step; explain options before choosing.",
            "逐步带我完成队列步骤；先解释选项，不要直接替我选择。",
            lang,
        ),
        _copilot_prompt_action(
            "route_fallback_modules",
            "Choose feature modules",
            "选择特征模块",
            "Choose feature modules.",
            "选择特征模块。",
            lang,
        ),
    ]


def _copilot_guided_choice_actions(
    study: Mapping[str, object],
    lang: str,
) -> list[dict[str, object]]:
    """Return in-chat choices for the current guided step."""
    step = str(study.get("step") or "question")
    question_substep = str(study.get("question_substep") or "")
    cohort_substep = str(study.get("cohort_substep") or "")
    route_actions = study.get("_route_actions")
    if isinstance(route_actions, list) and str(study.get("_route_actions_step") or "") == step:
        actions = [
            dict(action)
            for action in route_actions
            if isinstance(action, Mapping)
            and str(action.get("kind") or "") == "copilot_prompt"
            and str(action.get("label") or "").strip()
            and str(action.get("prompt") or "").strip()
        ]
        if actions:
            return actions[:5]
    if step == "question":
        if question_substep == "endpoint":
            return [
                _copilot_prompt_action(
                    "choice_endpoint_hospital_mortality",
                    "In-hospital mortality",
                    "院内死亡",
                    "endpoint: in-hospital mortality",
                    "endpoint：院内死亡",
                    lang,
                    desc_en="Death before hospital discharge — the most common ICU benchmark.",
                    desc_zh="出院前死亡——最常用的 ICU 基准结局。",
                ),
                _copilot_prompt_action(
                    "choice_endpoint_icu_mortality",
                    "ICU mortality",
                    "ICU 死亡",
                    "endpoint: ICU mortality",
                    "endpoint：ICU 死亡",
                    lang,
                    desc_en="Death within the ICU stay specifically.",
                    desc_zh="特指 ICU 住院期间的死亡。",
                ),
                _copilot_prompt_action(
                    "choice_endpoint_aki_rrt",
                    "AKI / RRT",
                    "AKI / RRT",
                    "endpoint: AKI or RRT",
                    "endpoint：AKI 或 RRT",
                    lang,
                    desc_en="Acute kidney injury onset or renal replacement therapy.",
                    desc_zh="急性肾损伤发生或肾脏替代治疗。",
                ),
                _copilot_prompt_action(
                    "choice_endpoint_custom",
                    "Type my endpoint",
                    "我自己输入 endpoint",
                    "I want to describe my own endpoint.",
                    "我想自己描述 endpoint。",
                    lang,
                    desc_en="Describe any other outcome in your own words.",
                    desc_zh="用自己的话描述任意其它结局。",
                ),
            ]
        if question_substep == "exposure":
            return [
                _copilot_prompt_action(
                    "choice_exposure_vasopressor",
                    "Vasopressors",
                    "升压药",
                    "exposure: vasopressor support",
                    "暴露：升压药支持",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_exposure_ventilation",
                    "Ventilation",
                    "机械通气",
                    "exposure: mechanical ventilation",
                    "暴露：机械通气",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_exposure_renal",
                    "RRT / renal support",
                    "RRT / 肾脏支持",
                    "exposure: renal replacement therapy",
                    "暴露：肾脏替代治疗",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_exposure_custom",
                    "Type exposure",
                    "我自己输入暴露",
                    "I want to describe my own exposure.",
                    "我想自己描述暴露。",
                    lang,
                ),
            ]
        if question_substep == "crossdb_signal":
            return [
                _copilot_prompt_action(
                    "choice_crossdb_outcome",
                    "Outcome signal",
                    "结局信号",
                    "cross-database signal: outcome model",
                    "跨库信号：结局模型",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_crossdb_treatment",
                    "Treatment pattern",
                    "治疗模式",
                    "cross-database signal: treatment pattern",
                    "跨库信号：治疗模式",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_crossdb_availability",
                    "Concept availability",
                    "概念可用性",
                    "cross-database signal: concept availability",
                    "跨库信号：概念可用性",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_crossdb_custom",
                    "Type signal",
                    "我自己输入信号",
                    "I want to describe my cross-database signal.",
                    "我想自己描述跨库信号。",
                    lang,
                ),
            ]
        if question_substep == "quality_target":
            return [
                _copilot_prompt_action(
                    "choice_quality_coverage",
                    "Coverage audit",
                    "覆盖率审计",
                    "quality target: concept coverage",
                    "质量目标：概念覆盖率",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_quality_mapping",
                    "Mapping / units",
                    "映射 / 单位",
                    "quality target: mapping and units",
                    "质量目标：映射和单位",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_quality_cohort",
                    "Cohort attrition",
                    "队列流失",
                    "quality target: cohort attrition",
                    "质量目标：队列流失",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_quality_custom",
                    "Type audit target",
                    "我自己输入审计目标",
                    "I want to describe my audit target.",
                    "我想自己描述审计目标。",
                    lang,
                ),
            ]
        return [
            _copilot_prompt_action(
                "choice_question_outcome_model",
                "Model an ICU outcome",
                "建模 ICU 结局",
                "question type: outcome prediction",
                "研究类型：结局预测",
                lang,
            ),
            _copilot_prompt_action(
                "choice_question_treatment_exposure",
                "Treatment exposure",
                "治疗暴露研究",
                "question type: treatment exposure",
                "研究类型：治疗暴露",
                lang,
            ),
            _copilot_prompt_action(
                "choice_question_crossdb",
                "Compare databases",
                "跨库比较",
                "question type: cross-database comparison",
                "研究类型：跨库比较",
                lang,
            ),
            _copilot_prompt_action(
                "choice_question_quality",
                "Audit data quality",
                "数据质量审计",
                "question type: data quality audit",
                "研究类型：数据质量审计",
                lang,
            ),
            _copilot_prompt_action(
                "choice_question_custom",
                "Type my question",
                "我自己描述问题",
                "I want to describe my own research question.",
                "我想自己描述研究问题。",
                lang,
            ),
        ]
    if step == "data":
        return [
            _copilot_prompt_action(
                "choice_data_prepared_path",
                "Prepared data path",
                "已有 prepared 路径",
                "I have a prepared data path.",
                "我有 prepared 数据路径。",
                lang,
                desc_en="A converted EasyICU folder (Parquet) — analysis-ready.",
                desc_zh="已转换的 EasyICU 文件夹(Parquet),可直接分析。",
            ),
            _copilot_prompt_action(
                "choice_data_module_export",
                "Module export folder",
                "已有模块导出",
                "I have an EasyICU module export folder.",
                "我有 EasyICU 模块导出文件夹。",
                lang,
                desc_en="A prior EasyICU export (per-concept Parquet + manifest).",
                desc_zh="之前的 EasyICU 导出(各概念 Parquet + manifest)。",
            ),
            _copilot_prompt_action(
                "choice_data_raw_files",
                "Raw ICU files",
                "只有 ICU 原始文件",
                "I only have raw ICU files.",
                "我只有 ICU 原始文件。",
                lang,
                desc_en="Original CSV/CSV.GZ download — needs conversion first.",
                desc_zh="原始 CSV/CSV.GZ 下载包,需先转换。",
            ),
        ]
    if step == "cohort":
        if cohort_substep == "disease":
            return [
                _copilot_prompt_action(
                    "choice_filter_sepsis",
                    "Sepsis-3",
                    "Sepsis-3",
                    "Filter cohort to Sepsis-3.",
                    "队列过滤为 Sepsis-3。",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_filter_aki",
                    "AKI / RRT",
                    "AKI / RRT",
                    "Filter cohort to AKI or RRT.",
                    "队列过滤为 AKI 或 RRT。",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_filter_none",
                    "No disease filter",
                    "不加疾病过滤",
                    "No disease filter.",
                    "不加疾病过滤。",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_filter_custom",
                    "Describe another",
                    "我自己描述",
                    "I want to describe another disease filter.",
                    "我想自己描述疾病过滤。",
                    lang,
                ),
            ]
        if cohort_substep == "age_los":
            return [
                _copilot_prompt_action(
                    "choice_filter_adult",
                    "Adult ICU stays",
                    "成人 ICU stay",
                    "Use adult ICU stays.",
                    "使用成人 ICU stay。",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_filter_los24",
                    "ICU LOS >= 24h",
                    "ICU LOS ≥ 24h",
                    "Require ICU LOS at least 24 hours.",
                    "要求 ICU LOS 至少 24 小时。",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_filter_broad",
                    "Keep broad",
                    "先保持宽松",
                    "No age or LOS restriction.",
                    "不加年龄或 LOS 限制。",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_filter_custom_age_los",
                    "Type constraints",
                    "我自己输入限制",
                    "I want to type age or ICU LOS constraints.",
                    "我想自己输入年龄或 ICU LOS 限制。",
                    lang,
                ),
            ]
        return [
            _copilot_prompt_action(
                "choice_cohort_eligible",
                "Eligible cohort",
                "全部合格队列",
                "Use the eligible cohort.",
                "使用全部合格队列。",
                lang,
                desc_en="Keep all analysis-eligible ICU stays — no clinical filter.",
                desc_zh="保留所有合格 ICU stay,不加临床过滤。",
            ),
            _copilot_prompt_action(
                "choice_cohort_disease",
                "Disease / diagnosis",
                "按疾病/诊断",
                "Configure a disease or diagnosis cohort filter.",
                "配置疾病或诊断队列过滤。",
                lang,
                desc_en="Add a clinical filter such as Sepsis-3, AKI, ARDS…",
                desc_zh="加临床过滤,如 Sepsis-3、AKI、ARDS……",
            ),
            _copilot_prompt_action(
                "choice_cohort_age_los",
                "Age / ICU LOS",
                "按年龄/ICU LOS",
                "Configure age or ICU length-of-stay constraints.",
                "配置年龄或 ICU LOS 限制。",
                lang,
                desc_en="Constrain demographics or minimum ICU stay length.",
                desc_zh="限制人口学或最短 ICU 住院时长。",
            ),
            _copilot_prompt_action(
                "choice_cohort_current",
                "Current reviewed cohort",
                "使用当前审阅队列",
                "Use the current reviewed cohort.",
                "使用当前审阅队列。",
                lang,
                desc_en="Reuse a cohort already loaded in Patient Review.",
                desc_zh="复用患者审阅里已加载的队列。",
            ),
        ]
    if step == "concepts":
        return _copilot_feature_module_actions(lang)
    if step in {"extract", "review", "analysis", "draft"}:
        if step == "extract":
            return [
                _copilot_prompt_action(
                    "choice_prepare_extraction",
                    "Prepare extraction plan",
                    "准备提取计划",
                    "Prepare the extraction plan in chat.",
                    "在聊天里准备提取计划。",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_explain_gate",
                    "Explain the gate",
                    "解释证据闸门",
                    "why this step?",
                    "为什么这一步？",
                    lang,
                ),
            ]
        if step == "review":
            return [
                _copilot_prompt_action(
                    "choice_run_review",
                    "Run visual review",
                    "运行可视化审阅",
                    "Run the visual review in chat.",
                    "在聊天里运行可视化审阅。",
                    lang,
                ),
                _copilot_prompt_action(
                    "choice_explain_gate",
                    "Explain the gate",
                    "解释证据闸门",
                    "why this step?",
                    "为什么这一步？",
                    lang,
                ),
            ]
        if step == "analysis":
            # Deterministic agent hand-off (no LLM route): seeds Research Agent
            # setup context + evidence gate. "Continue in chat" used to route
            # through the timeout-prone model here.
            return [
                {
                    "id": "agent_handoff",
                    "kind": "agent_handoff",
                    "label": "Set up agent run" if lang == "en" else "配置 Agent 运行",
                },
                _copilot_prompt_action(
                    "choice_explain_gate",
                    "Explain the gate",
                    "解释证据闸门",
                    "why this step?",
                    "为什么这一步？",
                    lang,
                ),
            ]
        return [
            _copilot_prompt_action(
                "choice_continue_chat",
                "Continue in chat",
                "继续聊天推进",
                "Continue this step in chat.",
                "继续在聊天里推进这一步。",
                lang,
            ),
            _copilot_prompt_action(
                "choice_explain_gate",
                "Explain the gate",
                "解释证据闸门",
                "why this step?",
                "为什么这一步？",
                lang,
            ),
        ]
    return []


def _copilot_capability_overview_actions(
    study: Mapping[str, object],
    state: Mapping[str, object],
    lang: str,
) -> list[dict[str, object]]:
    """Return high-level Copilot capabilities, adapted to the current progress."""
    has_session = bool(str(state.get("_copilot_current_session_dir") or "").strip())
    has_question = bool(str(study.get("question") or "").strip())
    data_status = str(study.get("data_source_status") or "").strip()
    has_data = data_status in {
        "pending_validation",
        "module_export_recorded",
        "conversion_needed",
    }
    has_cohort = bool(study.get("cohort_configured")) or _copilot_uses_eligible_cohort(study)
    has_modules = bool(study.get("concepts_configured")) or bool(study.get("selected_concepts"))

    actions: list[dict[str, object]] = []

    def add(action: dict[str, object]) -> None:
        if len(actions) >= 5:
            return
        if str(action.get("id") or "") not in {str(item.get("id") or "") for item in actions}:
            actions.append(action)

    if not has_session:
        add(_copilot_prompt_action(
            "capability_new_study",
            "New study / workspace",
            "新建研究 / 工作目录",
            "new study",
            "新研究",
            lang,
        ))
    if not has_question:
        add(_copilot_prompt_action(
            "capability_describe_goal",
            "Describe a study goal",
            "描述研究目标",
            "I want to describe my own research question.",
            "我想自己描述研究问题。",
            lang,
        ))
        add(_copilot_prompt_action(
            "capability_explore_ideas",
            "Explore research ideas",
            "探索研究 idea",
            "Recommend ICU research ideas in this chat, then ask me which one to inspect.",
            "在这个聊天里推荐 ICU 研究 idea，然后问我想继续看哪一个。",
            lang,
        ))
        add(_copilot_prompt_action(
            "capability_connect_data",
            "Connect real ICU data",
            "接入真实 ICU 数据",
            "Walk me through the real data source step and explain the prepared data path before choosing anything else.",
            "逐步带我完成真实数据源步骤；先解释 prepared 数据路径，不要直接选择其他部分。",
            lang,
        ))
        add(_copilot_prompt_action(
            "capability_full_flow",
            "Build a full workflow",
            "从完整流程开始",
            "Start a guided ICU study from the research question, then ask data source, cohort, modules, extraction, review, and Agent run one by one.",
            "从研究问题开始带我搭建完整 EasyICU 流程；再逐步问数据源、队列、模块、提取、审阅和 Agent 运行。",
            lang,
        ))
        return actions

    if not has_data:
        add(_copilot_prompt_action(
            "capability_connect_data",
            "Connect real ICU data",
            "接入真实 ICU 数据",
            "Walk me through the real data source step and explain the prepared data path before choosing anything else.",
            "逐步带我完成真实数据源步骤；先解释 prepared 数据路径，不要直接选择其他部分。",
            lang,
        ))
    if not has_cohort:
        add(_copilot_prompt_action(
            "capability_define_cohort",
            "Define the cohort",
            "定义研究队列",
            "Walk me through the cohort step; explain options before choosing.",
            "逐步带我完成队列步骤；先解释选项，不要直接替我选择。",
            lang,
        ))
    if not has_modules:
        add(_copilot_prompt_action(
            "capability_select_modules",
            "Select feature modules",
            "选择特征模块",
            "Choose feature modules.",
            "选择特征模块。",
            lang,
        ))
    add(_copilot_prompt_action(
        "capability_explore_ideas",
        "Explore related ideas",
        "探索相关 idea",
        "Recommend ICU research ideas in this chat, then ask me which one to inspect.",
        "在这个聊天里推荐 ICU 研究 idea，然后问我想继续看哪一个。",
        lang,
    ))
    if has_data and has_cohort and has_modules:
        add(_copilot_prompt_action(
            "capability_prepare_agent",
            "Prepare Agent run",
            "准备 Agent 分析",
            "Prepare the extraction plan in chat.",
            "在聊天里准备提取计划。",
            lang,
        ))
    add(_copilot_prompt_action(
        "capability_new_study",
        "New study / workspace",
        "新建研究 / 工作目录",
        "new study",
        "新研究",
        lang,
    ))
    return actions


def _copilot_hint_prompts(lang: str) -> list[tuple[str, str]]:
    """Return the polish guided composer hint chips and their submitted prompts."""
    if lang == "en":
        return [
            ("choose cohort", "Walk me through the cohort step; explain options before choosing."),
            ("why this step?", "why this step?"),
            ("go back", "go back"),
            ("set real data path", "what real data path should I use?"),
        ]
    return [
        ("选择队列", "逐步带我完成队列步骤；先解释选项，不要直接替我选择。"),
        ("为什么这一步？", "为什么这一步？"),
        ("返回上一步", "返回上一步"),
            ("设置真实路径", "真实数据路径应该填什么？"),
        ]


def _copilot_primary_prompts(lang: str) -> list[tuple[str, str, bool]]:
    """Return the top-level intent chips from the polish guided composer."""
    if lang == "en":
        return [
            (
                "Model ICU outcomes",
                "Start a guided ICU outcome study. Ask me to choose outcome, data source, cohort, and modules step by step.",
                False,
            ),
            (
                "Compare ICU databases",
                "Start a cross-database study. Walk me through cohort, outcome, databases, and feature checks one by one.",
                False,
            ),
            (
                "Explore research ideas",
                "Recommend ICU research ideas in this chat, then ask me which one to inspect.",
                False,
            ),
            (
                "Connect my own data",
                "Walk me through the real data source step and explain the prepared data path before choosing anything else.",
                False,
            ),
        ]
    return [
        (
            "建模 ICU 结局",
            "开始一个 ICU 结局研究向导。逐步问我选择结局、数据源、队列和模块。",
            False,
        ),
        (
            "比较 ICU 数据库",
            "开始一个跨数据库研究。逐步带我选择队列、结局、数据库和特征检查。",
            False,
        ),
        (
            "探索研究 idea",
            "在这个聊天里推荐 ICU 研究 idea，然后问我想继续看哪一个。",
            False,
        ),
        (
            "连接我的数据",
            "逐步带我完成真实数据源步骤；先解释 prepared 数据路径，不要直接选择其他部分。",
            False,
        ),
    ]


def _copilot_started_in_chat(state: Mapping[str, object], study: Mapping[str, object]) -> bool:
    """Return True after the user has entered a guided flow."""
    messages = state.get("llm_messages")
    return bool(
        (isinstance(messages, list) and messages)
        or str(study.get("branch") or "").strip()
        or str(study.get("question") or "").strip()
        or str(study.get("step") or "question") != "question"
        or bool(study.get("cohort_configured"))
        or bool(study.get("concepts_configured"))
    )


def _copilot_primary_actions_for_state(
    state: Mapping[str, object],
    lang: str,
) -> list[dict[str, object]]:
    """Return the main bottom chips for the user's current Copilot progress."""
    raw_study = state.get("_copilot_guided_study")
    study: Mapping[str, object] = raw_study if isinstance(raw_study, Mapping) else {}
    if not _copilot_started_in_chat(state, study):
        return _copilot_capability_overview_actions(study, state, lang)
    return _copilot_guided_choice_actions(study, lang)


def _copilot_hint_prompts_for_state(
    state: Mapping[str, object],
    lang: str,
) -> list[tuple[str, str]]:
    """Return secondary hint chips that follow the active guided step."""
    raw_study = state.get("_copilot_guided_study")
    study: Mapping[str, object] = raw_study if isinstance(raw_study, Mapping) else {}
    step = str(study.get("step") or "question")
    if not _copilot_started_in_chat(state, study):
        return (
            [
                ("what can I do?", "what can I do now?"),
                ("real data path", "what real data path should I use?"),
                ("why Copilot?", "how does this work?"),
                ("start over", "new study"),
            ]
            if lang == "en" else
            [
                ("我能做什么？", "我现在可以干什么"),
                ("真实数据路径", "真实数据路径应该填什么？"),
                ("为什么这样？", "这个怎么用？"),
                ("重新开始", "重新开始"),
            ]
        )
    if step == "data":
        return (
            [
                ("why data?", "why this step?"),
                ("go back", "go back"),
                ("set data path", "what real data path should I use?"),
                ("choose source", "I want to choose the data source type."),
            ]
            if lang == "en" else
            [
                ("为什么数据源？", "为什么这一步？"),
                ("返回研究问题", "返回上一步"),
                ("设置真实路径", "真实数据路径应该填什么？"),
                ("选择来源", "我想选择数据来源类型。"),
            ]
        )
    if step == "cohort":
        return (
            [
                ("why cohort?", "why this step?"),
                ("go back", "go back"),
                ("keep broad", "No disease filter."),
                ("set data path", "what real data path should I use?"),
            ]
            if lang == "en" else
            [
                ("为什么队列？", "为什么这一步？"),
                ("返回数据源", "返回上一步"),
                ("先保持宽松", "不加疾病过滤。"),
                ("设置真实路径", "真实数据路径应该填什么？"),
            ]
        )
    if step == "concepts":
        return (
            [
                ("why modules?", "why this step?"),
                ("go back", "go back"),
                ("use suggested", "use these modules"),
                ("set data path", "what real data path should I use?"),
            ]
            if lang == "en" else
            [
                ("为什么模块？", "为什么这一步？"),
                ("返回队列", "返回上一步"),
                ("使用推荐", "用这些变量"),
                ("设置真实路径", "真实数据路径应该填什么？"),
            ]
        )
    return _copilot_hint_prompts(lang)
