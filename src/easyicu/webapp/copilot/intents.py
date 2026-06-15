"""Research Copilot — prompt intent detection + lightweight text parsers.

Extracted from `llm_chat.py` (incremental Phase-6 split, see
`easyicu美化/copilot_接线施工计划.md` §6.1). These are pure string classifiers
(`_copilot_*_requested`, intent predicates) and small text→value parsers
(branch/database/concept/patient-count extraction). They take a prompt/text and
return bools or simple values — **no Streamlit `session_state` access** — so they
unit-test standalone and are safe to move without touching the test-suite's
`monkeypatch.setattr(llm_chat, "st", ...)` contract (which only matters for
session-state readers).

The few module-level `COPILOT_*` constants these need stay defined in
`llm_chat.py` and are imported lazily inside the using functions (same pattern as
`routing.py`), so this module never imports `llm_chat` at load time — no cycle.
`llm_chat.py` re-imports every name below, so all call sites and `llm_chat.<fn>`
test accesses keep working unchanged.
"""
from __future__ import annotations

import re

from easyicu.webapp.concept_catalog import CONCEPT_GROUPS_INTERNAL


def _infer_db_from_text(text: str) -> str | None:
    text_l = (text or "").lower()
    db_aliases = {
        "miiv": ("miiv", "mimic-iv", "mimic iv", "mimiciv"),
        "mimic": ("mimic-iii", "mimic iii", "mimiciii", "mimic 3"),
        "eicu": ("eicu", "eicu-crd"),
        "aumc": ("aumc", "amsterdamumcdb", "amsterdam umc"),
        "hirid": ("hirid",),
        "sic": ("sic", "sicdb"),
    }
    for db_key, aliases in db_aliases.items():
        if any(alias in text_l for alias in aliases):
            return db_key
    return None


def _copilot_pick_branch(text: str) -> str:
    text_l = (text or "").lower()
    if any(key in text_l for key in ("cross", "database", "databases", "replicate", "replication", "多库", "跨库", "数据库")):
        return "crossdb"
    if any(key in text_l for key in ("quality", "missing", "coverage", "audit", "sparse", "trust", "qc", "缺失", "质量", "覆盖")):
        return "quality"
    return "predict"


def _copilot_endpoint_pinned(text: str) -> bool:
    text_l = (text or "").lower()
    return bool(
        re.search(
            r"in-?hospital|28[\s-]*day|icu\s+mortality|icu\s+death|aki|rrt|renal|kidney|creatinine|urine output|length\s+of\s+stay|los|院内|28\s*天|icu\s*死亡|住院时长|肾脏替代|急性肾|肾损伤|肾损害|肾功能|肾衰|肌酐|尿量",
            text_l,
        )
    )


def _copilot_extract_patient_count(text: str) -> int | None:
    """Extract only explicit cohort-size commands, not labels like Sepsis-3."""
    text_l = (text or "").lower()
    patterns = [
        r"(?<![-/\w])(\d{1,6})\s*(?:demo\s*)?(?:icu\s*)?(?:patients?|cases?|stays?|subjects?)\b",
        r"(?<![-/\w])(\d{1,6})\s+(?:(?:demo|real|local|synthetic|icu|adult|sepsis-?3|sepsis|first)\s+){1,6}(?:patients?|cases?|stays?|subjects?)\b",
        r"(?<![-/\w])(\d{1,6})\s*(?:个|名)?\s*(?:患者|病例|例|人)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_l)
        if match:
            return max(5, min(100_000, int(match.group(1))))
    return None


def _copilot_real_data_requested(text: str) -> bool:
    text_l = (text or "").lower()
    return any(
        key in text_l
        for key in (
            "real data",
            "real-data",
            "local data",
            "local-data",
            "use local",
            "my data",
            "own data",
            "prepared data",
            "converted data",
            "真实数据",
            "本地数据",
            "我的数据",
            "实际数据",
            "已转换数据",
            "准备好的数据",
        )
    )


def _copilot_full_cohort_requested(text: str) -> bool:
    text_l = (text or "").lower()
    if (
        any(term in text_l for term in ("eligible", "full", "entire", "whole", "all"))
        and any(term in text_l for term in ("cohort", "stays", "patients", "subjects"))
    ):
        return True
    return any(
        key in text_l
        for key in (
            "eligible cohort",
            "eligible stays",
            "all eligible",
            "full cohort",
            "entire cohort",
            "whole cohort",
            "all patients",
            "all stays",
            "use the cohort",
            "formal analysis",
            "全量",
            "全部患者",
            "全部病例",
            "全部队列",
            "合格队列",
            "合格患者",
            "真实队列",
            "正式分析",
        )
    )


def _copilot_data_path_help_requested(text: str) -> bool:
    text_l = (text or "").lower()
    raw = text or ""
    path_terms = ("path", "data_path", "folder", "directory", "prepared", "converted", "路径", "目录", "文件夹")
    real_terms = (
        "real data",
        "real-data",
        "local data",
        "local-data",
        "my data",
        "own data",
        "mimic",
        "eicu",
        "hirid",
        "真实数据",
        "本地数据",
        "我的数据",
        "数据库",
        "这个路径",
    )
    return any(term in text_l or term in raw for term in path_terms) and any(
        term in text_l or term in raw for term in real_terms
    )


def _copilot_step_by_step_requested(text: str) -> bool:
    """Return True for starter prompts that should ask before choosing."""
    text_l = (text or "").lower()
    raw = text or ""
    return any(
        term in text_l
        for term in (
            "step by step",
            "one by one",
            "walk me through",
            "guided",
            "help me frame",
            "ask me",
            "do not choose",
            "don't choose",
            "do not decide",
            "don't decide",
            "before deciding",
        )
    ) or any(
        term in raw
        for term in (
            "逐步",
            "一步步",
            "向导",
            "先帮我框定",
            "先问我",
            "不要替我",
            "不要帮我决定",
            "再做决定",
        )
    )


def _copilot_cohort_step_requested(text: str) -> bool:
    """Return True when the user explicitly wants the cohort step."""
    text_l = (text or "").lower()
    raw = text or ""
    return (
        any(
            term in text_l
            for term in (
                "choose cohort",
                "cohort step",
                "cohort options",
                "configure cohort",
                "walk me through the cohort",
            )
        )
        or any(term in raw for term in ("选择队列", "队列步骤", "配置队列", "队列选项", "完成队列"))
    )


def _copilot_feature_step_requested(text: str) -> bool:
    """Return True when the user explicitly wants to open Step 3 module selection."""
    text_l = (text or "").lower()
    raw = text or ""
    if "add feature module:" in text_l or "feature module:" in text_l or "选择特征模块：" in raw:
        return False
    return any(
        term in text_l
        for term in (
            "choose feature modules",
            "feature module step",
            "module selection",
            "select modules",
            "configure modules",
        )
    ) or any(term in raw for term in ("选择特征模块", "特征模块步骤", "模块选择", "配置模块"))


def _copilot_extract_step_requested(text: str) -> bool:
    """Return True when the user (or a fixed step button) asks to run extraction.

    The "Prepare extraction plan" button submits a fixed prompt; routing that
    known intent through the LLM is fragile (a slow route times out and the step
    stalls). Detect it deterministically so extraction runs via the classic
    engine without any model round-trip.
    """
    text_l = (text or "").lower()
    raw = text or ""
    return any(
        term in text_l
        for term in (
            "prepare the extraction plan",
            "prepare extraction plan",
            "run the extraction",
            "run extraction",
            "extract the cohort",
            "extract now",
        )
    ) or any(term in raw for term in ("准备提取计划", "运行提取", "开始提取", "执行提取"))


def _copilot_review_step_requested(text: str) -> bool:
    """Return True when the user (or the fixed Review button) asks to run review.

    Same rationale as extraction: route the known intent deterministically to the
    classic engine instead of the timeout-prone LLM router.
    """
    text_l = (text or "").lower()
    raw = text or ""
    return any(
        term in text_l
        for term in (
            "run the visual review",
            "visual review in chat",
            "run the review",
            "review the cohort",
            "open the review",
        )
    ) or any(term in raw for term in ("运行审阅", "可视化审阅", "在聊天里审阅", "开始审阅"))


def _copilot_next_step_help_requested(text: str) -> bool:
    """Return True for broad "what do I do next" prompts that need local guidance."""
    text_l = (text or "").lower()
    raw = text or ""
    return any(
        term in text_l
        for term in (
            "what should i do",
            "what do i do",
            "what now",
            "what next",
            "next step",
            "where do i start",
            "how do i start",
            "how should i start",
        )
    ) or any(
        term in raw
        for term in (
            "我要做什么",
            "我该做什么",
            "现在做什么",
            "接下来做什么",
            "下一步",
            "下一步是什么",
            "从哪开始",
            "怎么开始",
        )
    )


def _copilot_usage_help_requested(text: str) -> bool:
    """Return True when the user asks how to use Copilot itself."""
    text_l = (text or "").lower()
    raw = text or ""
    return any(
        term in text_l
        for term in (
            "how do i use",
            "how to use",
            "how should i use",
            "how does this work",
            "why copilot",
            "why research copilot",
            "what can i do",
            "what can i do now",
            "what can this do",
            "what can copilot do",
            "getting started",
            "get started",
        )
    ) or any(
        term in raw
        for term in (
            "怎么使用",
            "如何使用",
            "怎么用",
            "如何用",
            "怎么开始用",
            "使用这个",
            "这个怎么用",
            "为什么这样",
            "为什么用 copilot",
            "可以干什么",
            "可以做什么",
            "能干什么",
            "能做什么",
            "能干嘛",
            "可以干嘛",
            "能帮我干什么",
            "能帮我做什么",
        )
    )


def _copilot_capability_overview_requested(text: str) -> bool:
    """Return True for broad capability questions, not current-step usage help."""
    text_l = (text or "").lower()
    raw = text or ""
    return any(
        term in text_l
        for term in (
            "what can i do",
            "what can i do now",
            "what can this do",
            "what can copilot do",
        )
    ) or any(
        term in raw
        for term in (
            "可以干什么",
            "可以做什么",
            "能干什么",
            "能做什么",
            "能干嘛",
            "可以干嘛",
            "能帮我干什么",
            "能帮我做什么",
        )
    )


def _copilot_extract_data_path_from_text(text: str) -> str:
    """Extract a local data/module path when the user configures it in chat."""
    raw = (text or "").strip()
    if not raw:
        return ""
    quoted = re.search(r"`([^`]+)`|[\"']([^\"']+)[\"']", raw)
    if quoted:
        candidate = str(quoted.group(1) or quoted.group(2) or "").strip()
        if candidate.startswith(("/", "~")):
            return candidate
    patterns = [
        r"(?:data[_ -]?path|prepared(?:/converted)?(?:\s+data)?\s+path|module\s+export(?:\s+folder)?|export\s+folder)\s*(?:is|=|:|为|是)?\s*(.+)$",
        r"(?:路径|目录|文件夹)\s*(?:是|为|=|:)?\s*(.+)$",
        r"((?:/|~)[^\n\r]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = str(match.group(1) or "").strip()
        candidate = re.sub(r"^(?:to|at|is|=|:|为|是)\s+", "", candidate, flags=re.IGNORECASE)
        candidate = candidate.strip(" \t\r\n'\"`.,;，。；")
        if candidate.startswith(("/", "~")):
            return candidate
    return ""


def _copilot_database_from_path(path: str, fallback: str = "miiv") -> str:
    path_l = (path or "").lower()
    if any(token in path_l for token in ("eicu", "eicu-crd")):
        return "eicu"
    if "hirid" in path_l:
        return "hirid"
    if "aumc" in path_l or "amsterdam" in path_l:
        return "aumc"
    if "mimiciii" in path_l or "mimic-iii" in path_l or "mimic3" in path_l:
        return "mimic"
    if any(token in path_l for token in ("miiv", "mimic-iv", "mimiciv")):
        return "miiv"
    if "sic" in path_l:
        return "sic"
    fallback_clean = str(fallback or "").strip()
    return "miiv" if fallback_clean == "mock" else (fallback_clean or "miiv")


def _copilot_normalize_database(database: object, fallback: str = "miiv") -> str:
    """Return a supported EasyICU database key for Copilot/classic state."""
    from easyicu.webapp.llm_chat import COPILOT_DATABASE_OPTIONS  # lazy: avoid import cycle
    raw = str(database or "").strip().lower()
    aliases = {
        "mimiciv": "miiv",
        "mimic-iv": "miiv",
        "mimic_iv": "miiv",
        "mimiciii": "mimic",
        "mimic-iii": "mimic",
        "mimic_iii": "mimic",
        "eicu-crd": "eicu",
        "umc": "aumc",
        "amsterdam": "aumc",
        "sicdb": "sic",
    }
    raw = aliases.get(raw, raw)
    if raw in COPILOT_DATABASE_OPTIONS:
        return raw
    fallback_clean = str(fallback or "miiv").strip().lower()
    return fallback_clean if fallback_clean in COPILOT_DATABASE_OPTIONS else "miiv"


def _copilot_feature_pack_label(pack_key: str, lang: str) -> str:
    from easyicu.webapp.llm_chat import COPILOT_FEATURE_MODULE_PACKS  # lazy: avoid import cycle
    pack = COPILOT_FEATURE_MODULE_PACKS.get(pack_key, {})
    return str(pack.get("label_en" if lang == "en" else "label_zh") or pack_key)


def _copilot_concepts_from_text(text: str) -> list[str]:
    """Map conversational concept names into EasyICU concept ids."""
    text_l = (text or "").lower()
    aliases = [
        (("lactate", "乳酸"), "lact"),
        (("sofa-2", "sofa2", "sofa"), "sofa2"),
        (("map", "mean arterial"), "map"),
        (("heart rate", "hr", "心率"), "hr"),
        (("temperature", "temp", "体温"), "temp"),
        (("spo2", "saturation", "氧饱和"), "spo2"),
        (("creatinine", "creat", "肌酐"), "crea"),
        (("urine", "urine output", "尿量"), "urine"),
        (("vasopressor", "norepi", "noradrenaline", "去甲", "升压"), "vaso_ind"),
        (("ventilation", "ventilator", "mechanical vent", "机械通气"), "vent_ind"),
        (("rrt", "renal replacement", "dialysis", "crrt", "透析", "肾替代"), "rrt"),
        (("age", "年龄"), "age"),
        (("mortality", "death", "死亡", "结局"), "death"),
    ]
    found: list[str] = []
    for terms, concept in aliases:
        if any(term in text_l for term in terms) and concept not in found:
            found.append(concept)
    return found


def _copilot_modules_for_concepts(concepts: list[str]) -> list[str]:
    """Return display modules that explain which classic feature groups are active."""
    concept_set = set(concepts)
    modules: list[str] = []
    for group_key, group_concepts in CONCEPT_GROUPS_INTERNAL.items():
        if concept_set.intersection(str(concept) for concept in group_concepts):
            modules.append(_copilot_feature_pack_label(group_key, "en"))
    return modules


def _copilot_is_strict_filter_request(text: str) -> bool:
    text_l = (text or "").lower()
    has_strict_word = any(
        key in text_l
        for key in ("restrict", "strict", "narrow", "tighten", "收紧", "严格", "限制")
    )
    has_age80 = bool(
        re.search(r"age\s*(?:>=|≥|over|older than|above)\s*80", text_l)
        or re.search(r"80\s*(?:years?|岁|以上)", text_l)
    )
    has_sepsis = any(key in text_l for key in ("sepsis-3", "sepsis 3", "sep3", "脓毒症"))
    return (has_strict_word and (has_age80 or has_sepsis)) or (has_age80 and has_sepsis)


def _copilot_is_loosen_filter_request(text: str) -> bool:
    text_l = (text or "").lower()
    return any(
        key in text_l
        for key in (
            "loosen",
            "back to defaults",
            "default filters",
            "defaults",
            "relax",
            "放宽",
            "恢复默认",
            "默认",
            "取消限制",
        )
    )


def _copilot_api_setup_requested(text: str) -> bool:
    text_l = (text or "").lower()
    return any(
        key in text_l
        for key in (
            "connect api",
            "api key",
            "openrouter",
            "model setup",
            "model settings",
            "llm settings",
            "token",
            "接入api",
            "接 api",
            "配置api",
            "模型设置",
            "模型配置",
            "连接模型",
        )
    )


def _copilot_route_next_question_asks_broad_type(text: str) -> bool:
    raw = text or ""
    text_l = raw.lower()
    return any(
        fragment in text_l
        for fragment in (
            "choose the research type",
            "choose study type",
            "select study type",
            "question type",
        )
    ) or any(
        fragment in raw
        for fragment in (
            "选择研究类型",
            "选择研究方向",
            "请选择研究类型",
            "请选择研究方向",
            "研究类型",
        )
    )


def _copilot_explicit_local_command(prompt: str) -> bool:
    raw = prompt or ""
    text_l = raw.lower()
    explicit_fragments = (
        "question type:",
        "endpoint:",
        "exposure:",
        "cross-database signal:",
        "quality target:",
        "set data path",
        "prepared data path",
        "prepared path",
        "module export",
        "raw icu files",
        "raw files",
        "add feature module:",
        "feature module:",
        "choose feature modules",
        "use these modules",
        "use suggested modules",
        "use eligible cohort",
        "filter cohort to",
        "describe my own research question",
        "describe my own endpoint",
        "describe my own exposure",
        "describe my cross-database signal",
        "describe my audit target",
        "current reviewed cohort",
        "no disease filter",
        "why this step",
        "go back",
        "run the whole demo",
        "whole demo",
    )
    explicit_zh = (
        "研究类型：",
        "endpoint：",
        "暴露：",
        "跨库信号：",
        "质量目标：",
        "真实数据路径",
        "prepared 数据路径",
        "已有 prepared",
        "模块导出",
        "原始文件",
        "选择特征模块",
        "用这些变量",
        "全部合格队列",
        "队列过滤为",
        "自己描述研究问题",
        "自己描述 endpoint",
        "自己描述暴露",
        "自己描述跨库信号",
        "自己描述审计目标",
        "当前审阅队列",
        "不加疾病过滤",
        "为什么这一步",
        "返回上一步",
        "跑完整演示",
    )
    return any(fragment in text_l for fragment in explicit_fragments) or any(fragment in raw for fragment in explicit_zh)


def _copilot_free_study_goal_requested(prompt: str) -> bool:
    raw = (prompt or "").strip()
    if len(raw) < 6:
        return False
    if _copilot_explicit_local_command(raw):
        return False
    text_l = raw.lower()
    workflow_terms = (
        "study",
        "research",
        "analysis",
        "analyze",
        "model",
        "predict",
        "cluster",
        "trajectory",
        "compare",
        "audit",
        "cohort",
        "endpoint",
        "outcome",
    )
    workflow_terms_zh = (
        "研究",
        "分析",
        "建模",
        "预测",
        "聚类",
        "轨迹",
        "比较",
        "审计",
        "队列",
        "结局",
        "预警",
        "我要做",
        "我想做",
    )
    return any(term in text_l for term in workflow_terms) or any(term in raw for term in workflow_terms_zh)


def _copilot_should_use_llm_route(
    prompt: str,
    *,
    usage_help_intent: bool,
    step_by_step_intent: bool,
    full_cohort_intent: bool = False,
    cohort_step_intent: bool,
    api_intent: bool,
    path_help_intent: bool,
    guided_choice_intent: bool,
) -> bool:
    if usage_help_intent:
        return False
    if api_intent:
        return False
    if step_by_step_intent:
        return False
    if full_cohort_intent:
        return False
    if cohort_step_intent:
        return False
    if path_help_intent:
        return False
    if guided_choice_intent or _copilot_explicit_local_command(prompt):
        return False
    if _copilot_extract_patient_count(prompt) is not None:
        return False
    if _copilot_extract_data_path_from_text(prompt):
        return False
    return len((prompt or "").strip()) >= 3 or _copilot_free_study_goal_requested(prompt)


def _copilot_first_pass_goal_allowed(prompt: str) -> bool:
    """Return True only for natural-language study goals, not workflow controls."""
    raw = (prompt or "").strip()
    text_l = raw.lower()
    if not _copilot_free_study_goal_requested(raw):
        return False
    if _copilot_explicit_local_command(raw):
        return False
    if _copilot_extract_patient_count(raw) is not None:
        return False
    if _copilot_extract_data_path_from_text(raw):
        return False
    control_prefixes = (
        "use ",
        "set ",
        "choose ",
        "select ",
        "configure ",
        "save ",
        "run ",
        "open ",
        "walk me through",
    )
    if text_l.startswith(control_prefixes):
        return False
    open_goal_markers_en = (
        "i want",
        "i would like",
        "i need",
        "i'm interested",
        "help me",
    )
    open_goal_markers_zh = ("我想", "我要", "想做", "要做", "帮我")
    return any(marker in text_l for marker in open_goal_markers_en) or any(
        marker in raw for marker in open_goal_markers_zh
    )


def _copilot_prepared_data_choice_requested(text: str) -> bool:
    text_l = (text or "").lower()
    raw = text or ""
    return any(term in text_l for term in ("prepared data path", "prepared path")) or any(
        term in raw for term in ("prepared 数据路径", "已有 prepared 路径", "已有 prepared 数据")
    )


def _copilot_module_export_choice_requested(text: str) -> bool:
    text_l = (text or "").lower()
    raw = text or ""
    return "module export" in text_l or any(term in raw for term in ("模块导出", "导出文件夹"))


def _copilot_raw_files_choice_requested(text: str) -> bool:
    text_l = (text or "").lower()
    raw = text or ""
    return "raw icu files" in text_l or "raw files" in text_l or any(
        term in raw for term in ("原始文件", "原始数据", "只有 ICU 原始")
    )


def _copilot_disease_cohort_choice_requested(text: str) -> bool:
    text_l = (text or "").lower()
    raw = text or ""
    return "disease or diagnosis cohort" in text_l or any(
        term in raw for term in ("疾病或诊断队列", "疾病/诊断", "诊断队列")
    )


def _copilot_age_los_choice_requested(text: str) -> bool:
    text_l = (text or "").lower()
    raw = text or ""
    return "age or icu length-of-stay" in text_l or "age / icu los" in text_l or any(
        term in raw for term in ("年龄或 ICU LOS", "年龄/ICU LOS", "ICU LOS 限制")
    )


def _copilot_current_reviewed_cohort_requested(text: str) -> bool:
    text_l = (text or "").lower()
    raw = text or ""
    return "current reviewed cohort" in text_l or any(term in raw for term in ("当前审阅队列", "当前队列"))


def _copilot_no_disease_filter_requested(text: str) -> bool:
    text_l = (text or "").lower()
    raw = text or ""
    return "no disease filter" in text_l or any(term in raw for term in ("不加疾病过滤", "不按疾病过滤", "无疾病过滤"))


def _copilot_module_pack_from_prompt(text: str) -> list[str] | None:
    from easyicu.webapp.llm_chat import COPILOT_FEATURE_MODULE_PACKS  # lazy: avoid import cycle
    text_l = (text or "").lower()
    raw = text or ""
    requested = ""
    if "add feature module:" in text_l:
        requested = text_l.split("add feature module:", 1)[1]
    elif "feature module:" in text_l:
        requested = text_l.split("feature module:", 1)[1]
    elif "选择特征模块：" in raw:
        requested = raw.split("选择特征模块：", 1)[1]
    if requested:
        requested_clean = requested.strip().strip("。.!? ").lower()
        for key, pack in COPILOT_FEATURE_MODULE_PACKS.items():
            labels = {
                key.lower().replace("_", " "),
                str(pack.get("label_en") or "").lower(),
                str(pack.get("label_zh") or "").lower(),
            }
            if requested_clean in labels:
                return [str(concept) for concept in pack["concepts"]]

    # Backward compatibility for buttons stored in older local sessions. New UI
    # renders real Step 3 module labels from CONCEPT_GROUPS_INTERNAL instead.
    if "core bedside modules" in text_l or "核心床旁" in raw:
        return [str(concept) for concept in COPILOT_FEATURE_MODULE_PACKS["vitals"]["concepts"]]
    if "severity score" in text_l or "严重程度评分" in raw:
        return [str(concept) for concept in COPILOT_FEATURE_MODULE_PACKS["sofa2_score"]["concepts"]]
    if "labs and treatment" in text_l or "labs + treatment" in text_l or "化验和治疗" in raw or "化验 + 治疗" in raw:
        concepts: list[str] = []
        for key in ("chemistry", "blood_gas", "vasopressors", "respiratory"):
            for concept in COPILOT_FEATURE_MODULE_PACKS[key]["concepts"]:
                if str(concept) not in concepts:
                    concepts.append(str(concept))
        return concepts
    return None


def _copilot_research_recommendation_requested(text: str) -> bool:
    """Return True when the user asks Copilot to recommend research directions."""
    raw = text or ""
    text_l = raw.lower()
    recommend_terms_en = (
        "recommend",
        "suggest",
        "what should i study",
        "research direction",
        "study direction",
        "study idea",
        "topic idea",
        "any ideas",
    )
    recommend_terms_zh = (
        "推荐",
        "建议",
        "有什么推荐",
        "有啥推荐",
        "研究方向",
        "选题",
        "课题",
        "做什么研究",
        "有什么方向",
    )
    research_terms_en = ("research", "study", "topic", "idea", "hypothesis", "icu", "sepsis")
    research_terms_zh = ("研究", "选题", "课题", "方向", "icu", "ICU", "重症", "脓毒症")
    return (
        any(term in text_l for term in recommend_terms_en)
        and any(term in text_l for term in research_terms_en)
    ) or (
        any(term in raw for term in recommend_terms_zh)
        and any(term in raw for term in research_terms_zh)
    )


def _copilot_idea_topic(text: str) -> str:
    text_l = (text or "").lower()
    if any(term in text_l for term in ("sepsis", "sepsis-3", "septic")) or "脓毒症" in str(text or ""):
        return "sepsis"
    return "general"


def _copilot_parse_idea_selection(text: str) -> int | None:
    raw = (text or "").strip()
    text_l = raw.lower()
    if re.fullmatch(r"[123]", text_l):
        return int(text_l) - 1
    patterns = [
        r"(?:pick|choose|select|use|go with|continue)\s*(?:option\s*)?([123])",
        r"(?:选|选择|继续|用|就用|第)\s*([123])",
        r"第\s*([123])\s*个",
    ]
    for pattern in patterns:
        match = re.search(pattern, text_l)
        if match:
            return int(match.group(1)) - 1
    chinese_numbers = {"一": 0, "二": 1, "三": 2}
    for token, idx in chinese_numbers.items():
        if re.search(rf"(?:选|选择|继续|用|就用|第)\s*{token}\s*个?", raw):
            return idx
    return None


def _copilot_confirm_suggested_concepts_requested(text: str) -> bool:
    text_l = (text or "").lower()
    raw = text or ""
    return any(
        term in text_l
        for term in ("use these modules", "use these concepts", "confirm modules", "accept modules")
    ) or any(term in raw for term in ("用这些变量", "使用这些变量", "确认变量", "确认模块", "就这些模块"))
