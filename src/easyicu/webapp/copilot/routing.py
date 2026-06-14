"""Research Copilot — offline keyword routing + disease/sepsis option helpers.

Extracted from `llm_chat.py` (incremental Phase-6 split). These are pure helpers
with no Streamlit session-state dependency, so they unit-test standalone
(`tests/webapp/test_copilot_keyword_route.py`).

The constants `COPILOT_DISEASE_OPTIONS` / `COPILOT_ROUTE_FAMILY_LABELS` stay in
`llm_chat.py` and are imported lazily inside the functions to avoid an import
cycle (llm_chat imports this module at load time; this module only needs those
constants at call time, when llm_chat is already fully loaded).
"""
from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping

from easyicu.webapp.copilot.llm import _strip_llm_reasoning


def _copilot_full_disease_options(lang: str) -> "dict[str, str]":
    """Full classic disease-cohort list (curated 6 + ICD-based templates).

    Tier-3 advanced disclosure: the chat dropdown surfaces every template the
    classic Step 2 supports (DISEASE_COHORT_CONFIG) so power users never have to
    leave the conversation. Writes still flow through the same classic engine
    (`apply_cohort_filter`), so the chat cohort stays byte-identical to classic.
    """
    from easyicu.webapp.llm_chat import COPILOT_DISEASE_OPTIONS

    options: dict[str, str] = {
        key: (label_en if lang == "en" else label_zh)
        for key, (label_en, label_zh) in COPILOT_DISEASE_OPTIONS.items()
    }
    try:
        from easyicu.webapp.cohort_config import DISEASE_COHORT_CONFIG
    except Exception:
        return options
    for key, cfg in DISEASE_COHORT_CONFIG.items():
        if key in options:
            continue
        options[key] = str(
            cfg.get("label_en" if lang == "en" else "label_zh") or key
        )
    return options


def _copilot_sepsis_mode_options(lang: str) -> "dict[str, str]":
    """Sepsis suspected-infection modes (classic `sepsis_si_mode`)."""
    try:
        from easyicu.webapp.cohort_config import SEPSIS_MODE_CONFIG
    except Exception:
        return {"auto": "Auto by database" if lang == "en" else "按数据库自动选择"}
    return {
        key: str(cfg.get("label_en" if lang == "en" else "label_zh") or key)
        for key, cfg in SEPSIS_MODE_CONFIG.items()
    }


def _copilot_keyword_route(prompt: str, lang: str) -> "dict[str, object] | None":
    """Deterministic offline fallback for the free-text study goal.

    When no model is reachable, classify the goal with light keyword rules into
    the SAME route schema `_copilot_apply_llm_route` consumes, so the natural-
    language entry still flows without an API key. Returns None when the text is
    too vague to classify (the caller then shows the model/clarify fallback).

    This is an explicit, opt-in degradation (the LLM is still tried first); when
    a model is configured it always wins. Writes still go through the shared
    applier, so chat and classic stay consistent.
    """
    from easyicu.webapp.llm_chat import COPILOT_ROUTE_FAMILY_LABELS

    raw = (prompt or "").strip()
    if len(raw) < 8:
        return None
    t = raw.lower()

    db_names = ("mimic", "eicu", "aumc", "hirid", "sicdb", "amsterdam", "salzburg")
    db_mentions = sum(1 for d in db_names if d in t)
    compare_word = any(
        w in t for w in (
            "compare", "comparison", "across", "versus", " vs ", "vs.",
            "benchmark", "multi-center", "multi-centre", "multi-site",
        )
    )
    if (
        any(k in t for k in (
            "cross-database", "cross database", "across databases",
            "multi-database", "multiple databases", "compare database",
            "compare databases", "external valid", "externally valid", "replicat",
            "generaliz", "transport",
        ))
        # "compare … across … databases" with words in between, or ≥2 DB names.
        or (compare_word and "database" in t)
        or db_mentions >= 2
        or any(k in raw for k in ("跨库", "跨数据库", "多数据库", "比较数据库", "外部验证", "多中心", "泛化"))
    ):
        family = "cross_database"
    elif (
        any(k in t for k in (
            "data quality", "missingness", "missing data", "harmoniz",
            "completeness", "audit",
        ))
        or any(k in raw for k in ("数据质量", "缺失", "审计", "质量", "完整性", "一致性"))
    ):
        family = "quality_audit"
    elif (
        any(k in t for k in ("associat", "risk factor", "correlat", "relationship between"))
        or any(k in raw for k in ("关联", "相关", "危险因素", "关系"))
    ):
        family = "association"
    else:
        family = "prediction"

    label_en, label_zh = COPILOT_ROUTE_FAMILY_LABELS.get(
        family, COPILOT_ROUTE_FAMILY_LABELS["unknown"]
    )
    label = label_en if lang == "en" else label_zh

    route: dict[str, object] = {
        "analysis_family": family,
        "analysis_label": label,
        "study_frame": raw,
        "current_step": "question" if family == "prediction" else "data",
    }

    disease_kw = {
        "sepsis": ("sepsis", "septic", "脓毒", "脓毒症"),
        "aki": ("aki", "acute kidney", "肾损伤", "急性肾"),
        "ards": ("ards", "acute respiratory distress", "呼吸窘迫"),
        "pneumonia": ("pneumonia", "肺炎"),
        "heart_failure": ("heart failure", "心衰", "心力衰竭"),
        "ami": ("myocardial infarction", "心肌梗死", "心梗"),
        "stroke": ("stroke", "卒中", "脑卒中", "中风"),
    }
    disease_labels = _copilot_full_disease_options(lang)
    for key, kws in disease_kw.items():
        if any(kw in t for kw in kws) or any(kw in raw for kw in kws):
            route["cohort"] = {"label": disease_labels.get(key, key)}
            break

    concept_kw = {
        "lact": ("lactate", "乳酸"),
        "sofa2": ("sofa", "sofa-2", "sofa2"),
        "creat": ("creatinine", "肌酐"),
        "map": ("mean arterial", "blood pressure", "血压"),
        "hr": ("heart rate", "心率"),
        "spo2": ("spo2", "oxygen satur", "血氧"),
    }
    concepts = [
        key for key, kws in concept_kw.items()
        if any(kw in t for kw in kws) or any(kw in raw for kw in kws)
    ]
    if concepts:
        route["suggested_concepts"] = concepts

    # Guard against over-eager classification: only classify when there is an
    # actual clinical/research signal. Otherwise (greeting, off-topic, vague)
    # return None so the caller shows the clarify/model fallback instead of
    # silently framing it as a prediction study.
    explicit_family = family in {"cross_database", "quality_audit", "association"}
    clinical_terms_en = (
        "predict", "prognos", "mortality", "death", "died", "outcome", "risk",
        "survival", "length of stay", " los ", "readmission", "icu", "patient",
        "cohort", "ventilat", "vasopressor", "dialysis", "rrt", "delirium",
        "shock", "organ failure", "sofa", "sepsis", "kidney", "respiratory",
    )
    clinical_terms_zh = (
        "预测", "预后", "死亡", "结局", "风险", "生存", "住院时长", "再入院",
        "机械通气", "升压药", "透析", "谵妄", "休克", "器官衰竭", "患者",
        "队列", "重症", "脓毒", "肾", "呼吸",
    )
    has_signal = (
        explicit_family
        or "cohort" in route
        or bool(concepts)
        or any(term in t for term in clinical_terms_en)
        or any(term in raw for term in clinical_terms_zh)
    )
    if not has_signal:
        return None

    prefix = (
        "Routed locally without a model (configure API for richer parsing). "
        if lang == "en" else
        "未连接模型,已用本地规则判断路线(配置 API 可获得更精准解析)。"
    )
    route["assistant_text"] = prefix + (
        f"I read this as **{label}**."
        if lang == "en" else
        f"我把它理解为 **{label}**。"
    )
    return route


# --- Phase-6 split (6th batch): LLM-route choice/parse helpers --------------
# Pure helpers (no Streamlit `st`) that build route-choice action lists, classify
# route specificity, format the route-family label, run the bounded route
# completion call, and parse the route JSON. The `st`-touching transport
# (`_copilot_route_with_llm`, `_copilot_route_transport_enabled`) stays in
# llm_chat. `COPILOT_ROUTE_*` constants are lazy-imported (see module docstring).


def _copilot_route_family_label(family: str, lang: str) -> str:
    from easyicu.webapp.llm_chat import COPILOT_ROUTE_FAMILY_LABELS  # lazy
    label_en, label_zh = COPILOT_ROUTE_FAMILY_LABELS.get(
        family,
        COPILOT_ROUTE_FAMILY_LABELS["unknown"],
    )
    return label_en if lang == "en" else label_zh


def _copilot_branch_for_route_family(family: str) -> str:
    if family == "cross_database":
        return "crossdb"
    if family == "quality_audit":
        return "quality"
    return "predict"


def _copilot_sanitize_route_choice_id(value: str, fallback: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", (value or "").strip().lower()).strip("_")
    return token[:48] or fallback


def _copilot_route_choice_actions(route: Mapping[str, object], lang: str) -> list[dict[str, object]]:
    raw_choices = route.get("choices")
    if not isinstance(raw_choices, list):
        return []
    actions: list[dict[str, object]] = []
    for idx, raw_choice in enumerate(raw_choices[:5], start=1):
        if not isinstance(raw_choice, Mapping):
            continue
        label = str(raw_choice.get("label") or raw_choice.get("title") or "").strip()
        prompt = str(raw_choice.get("prompt") or label).strip()
        if not label or not prompt:
            continue
        raw_id = str(raw_choice.get("id") or label)
        action_id = "route_choice_" + _copilot_sanitize_route_choice_id(raw_id, f"choice_{idx}")
        if any(action["id"] == action_id for action in actions):
            action_id = f"{action_id}_{idx}"
        actions.append({
            "id": action_id,
            "kind": "copilot_prompt",
            "label": label[:80],
            "prompt": prompt[:600],
        })
    return actions


def _copilot_route_has_specific_goal(route: Mapping[str, object]) -> bool:
    family = str(route.get("analysis_family") or "").strip().lower()
    frame = str(route.get("study_frame") or route.get("question") or "").strip()
    label = str(route.get("analysis_label") or "").strip()
    return bool(frame) or (family and family != "unknown") or bool(label)


def _copilot_route_uses_broad_question_type_choices(actions: list[dict[str, object]]) -> bool:
    if not actions:
        return False
    broad_prompts = ("question type:", "研究类型：")
    broad_labels = {
        "model an icu outcome",
        "建模 icu 结局",
        "treatment exposure",
        "治疗暴露研究",
        "compare databases",
        "跨库比较",
        "audit data quality",
        "数据质量审计",
    }
    broad_count = 0
    for action in actions:
        label = str(action.get("label") or "").strip().lower()
        prompt = str(action.get("prompt") or "").strip().lower()
        if label in broad_labels or any(fragment in prompt for fragment in broad_prompts):
            broad_count += 1
    if len(actions) <= 2:
        return broad_count >= 1
    return broad_count >= max(2, min(len(actions), 3))


def _copilot_specific_route_next_question(family: str, step: str, lang: str) -> str:
    if family == "prediction" and step == "question":
        return (
            "First choose the event or endpoint you want the warning model to predict; you can also type your own."
            if lang == "en" else
            "先确认你要预警的事件或 endpoint；可以直接选择，也可以自己输入。"
        )
    if family == "clustering":
        return (
            "Next, confirm the data source and the feature space for clustering in this same chat."
            if lang == "en" else
            "下一步在当前聊天里确认数据源和用于聚类的特征空间。"
        )
    return (
        "Choose the next concrete workflow item below; we will stay in this Copilot page."
        if lang == "en" else
        "请在下面选择下一个具体工作流项；我们仍然留在当前 Copilot 页面。"
    )


def _copilot_create_route_completion(route_client: object, request_kwargs: dict[str, object]) -> object | None:
    """Call the route model with a hard wall-clock timeout.

    Some OpenAI-compatible endpoints do not enforce the client timeout as a
    strict UI budget. The Copilot route must never block the Streamlit rerun
    longer than the interaction budget, so the request runs in a daemon thread.
    """
    from easyicu.webapp.llm_chat import COPILOT_ROUTE_TIMEOUT_SECONDS  # lazy
    result: dict[str, object] = {}

    def _call() -> None:
        try:
            result["response"] = route_client.chat.completions.create(**request_kwargs)
        except Exception as exc:  # noqa: BLE001 - returned to the caller as route unavailable
            result["error"] = exc

    worker = threading.Thread(target=_call, daemon=True)
    worker.start()
    worker.join(COPILOT_ROUTE_TIMEOUT_SECONDS)
    if worker.is_alive():
        return None
    if result.get("error") is not None:
        raise result["error"]  # type: ignore[misc]
    return result.get("response")


def _copilot_extract_route_json(text: str) -> dict[str, object] | None:
    cleaned = _strip_llm_reasoning(text or "").strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None
