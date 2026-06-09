"""Research Copilot submit / apply / step-execution workflow (batch-11 split from llm_chat.py).

The conversational submit handlers and the ``_copilot_run_step`` dispatcher.
Every function operates on the explicit ``state`` / ``study`` / ``app_context``
arguments it is handed and routes state-writes through the classic engines
(via in-body lazy imports), so they hold **zero ``st`` reference**
(AST-verified) and live outside ``llm_chat`` without breaking the
``monkeypatch.setattr(llm_chat, "st", fake)`` test contract. ``llm_chat``
re-imports every name so call sites, parity tests, and the engine dispatcher
keep using ``llm_chat.<fn>`` unchanged.
"""

from __future__ import annotations

import re  # noqa: F401  (used in handlers)
from collections.abc import Mapping, MutableMapping  # noqa: F401  (annotations)
from datetime import datetime  # noqa: F401
from pathlib import Path  # noqa: F401

from easyicu.webapp import copilot_engine as _copilot_engine
from easyicu.webapp.copilot.actions import (
    _copilot_guided_choice_actions,
    _copilot_prompt_action,
)
from easyicu.webapp.copilot.handoff import _apply_copilot_study_to_workspace
from easyicu.webapp.copilot.idea import _copilot_reply
from easyicu.webapp.copilot.intents import (
    _copilot_concepts_from_text,
    _copilot_database_from_path,
    _copilot_extract_patient_count,
    _copilot_feature_pack_label,
    _copilot_modules_for_concepts,
    _copilot_normalize_database,
)
from easyicu.webapp.copilot.presentation import (
    _copilot_concept_label_list,
    _copilot_data_source_choice_label,
    _copilot_disease_label,
)
from easyicu.webapp.copilot.routing import (
    _copilot_full_disease_options,
    _copilot_sepsis_mode_options,
)
from easyicu.webapp.copilot.sessions import _ensure_copilot_study_state

def _copilot_default_cohort_filter() -> dict[str, object]:
    """Mirror the classic Step 2 cohort filter schema without importing sidebar."""
    return {
        "age_min": None,
        "age_max": None,
        "first_icu_stay": None,
        "los_min": None,
        "los_max": None,
        "gender": None,
        "survived": None,
        "has_sepsis": None,
        "disease_cohort": "none",
        "icd_query": "",
        "icd_include_query": "",
        "icd_exclude_query": "",
        "icd_mode": "include",
    }


def _copilot_confirm_classic_step2(state: MutableMapping[str, object]) -> None:
    """Mirror a Copilot cohort choice into the classic extraction gate."""
    state["step2_confirmed"] = True
    state["step3_confirmed"] = False
    state["export_completed"] = False


def _copilot_confirm_classic_step3(state: MutableMapping[str, object]) -> None:
    """Mirror a Copilot module choice into the classic extraction gate."""
    state["step3_confirmed"] = True
    state["export_completed"] = False


def _copilot_parse_optional_int(value: object) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = int(float(text))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _copilot_validate_real_data_path(state: MutableMapping[str, object]) -> bool:
    """Run the SAME validator the classic Data Source step uses.

    The classic flow requires Validate Data Path + Confirm before extraction can
    read the prepared tables; the chat flow previously recorded a path but left
    ``path_validated`` False, so the loader found nothing ("0 tables"). This runs
    `validate_database_path` and, on success, marks the source validated/confirmed
    exactly like the classic Step 1, so chat extraction matches classic.
    """
    data_path = str(state.get("data_path") or "").strip()
    database = _copilot_normalize_database(state.get("database") or "miiv")
    if not data_path:
        state["path_validated"] = False
        return False
    try:
        from easyicu.webapp.data_workflows import validate_database_path

        result = validate_database_path(data_path, database, state.get("_app_context"))
    except Exception:
        state["path_validated"] = False
        return False
    valid = bool(isinstance(result, Mapping) and result.get("valid"))
    state["path_validated"] = valid
    if valid:
        state["last_validated_path"] = data_path
        state["step1_confirmed"] = True
    return valid


def _copilot_set_real_data_path_in_chat(
    state: MutableMapping[str, object],
    path: str,
    database: str | None = None,
) -> None:
    """Bind a typed real-data path without leaving the Copilot chat page."""
    clean_path = str(path or "").strip()
    if not clean_path:
        return
    state["entry_mode"] = "real"
    state["use_mock_data"] = False
    state["data_path"] = clean_path
    state["database"] = _copilot_database_from_path(
        clean_path,
        _copilot_normalize_database(database or state.get("database") or "miiv"),
    )
    state.pop("last_validated_path", None)
    state["sidebar_data_path_input__pending_value"] = clean_path
    study = _ensure_copilot_study_state(state)
    study["data_mode"] = "real"
    study["step"] = "data"
    study["data_source_choice"] = "prepared_path"
    study["data_source_path_label"] = Path(clean_path).name or clean_path
    study["database"] = state["database"]
    # Validate immediately (mirrors classic Validate + Confirm) so the recorded
    # path is actually usable for extraction, not just stored.
    validated = _copilot_validate_real_data_path(state)
    study["data_source_status"] = "pending_validation" if validated else "conversion_needed"
    # D3: data source changed → cohort stats + extraction/review are stale.
    state.pop("_cohort_stats", None)
    state.pop("_cohort_filtered_ids", None)
    state.pop("filtered_patient_count", None)
    _copilot_invalidate_downstream_extraction(state, study)
    study["last_update"] = datetime.now().isoformat(timespec="seconds")


def _copilot_set_module_export_path_in_chat(
    state: MutableMapping[str, object],
    path: str,
    database: str | None = None,
) -> None:
    """Bind an EasyICU module export folder from the Copilot page."""
    clean_path = str(path or "").strip()
    if not clean_path:
        return
    state["entry_mode"] = "real"
    state["use_mock_data"] = False
    state["database"] = _copilot_normalize_database(database or state.get("database") or "miiv")
    state["last_export_dir"] = clean_path
    state["export_path"] = clean_path
    study = _ensure_copilot_study_state(state)
    study["data_mode"] = "real"
    study["step"] = "data"
    study["data_source_choice"] = "module_export"
    study["data_source_status"] = "module_export_recorded"
    study["data_source_path_label"] = Path(clean_path).name or clean_path
    study["database"] = state["database"]
    study["last_update"] = datetime.now().isoformat(timespec="seconds")


def _copilot_set_raw_icu_path_in_chat(
    state: MutableMapping[str, object],
    path: str,
    database: str | None = None,
) -> None:
    """Record a raw ICU root folder without leaving Copilot."""
    clean_path = str(path or "").strip()
    if not clean_path:
        return
    clean_database = _copilot_normalize_database(database or state.get("database") or "miiv")
    state["entry_mode"] = "real"
    state["use_mock_data"] = False
    state["database"] = _copilot_database_from_path(clean_path, clean_database)
    state["raw_data_path"] = clean_path
    state["data_path"] = clean_path
    state["path_validated"] = False
    state.pop("last_validated_path", None)
    state["sidebar_data_path_input__pending_value"] = clean_path
    study = _ensure_copilot_study_state(state)
    study["data_mode"] = "real"
    study["step"] = "data"
    study["data_source_choice"] = "raw_files"
    study["data_source_status"] = "conversion_needed"
    study["data_source_path_label"] = Path(clean_path).name or clean_path
    study["database"] = state["database"]
    study["last_update"] = datetime.now().isoformat(timespec="seconds")


def _copilot_submit_data_source_path(
    state: MutableMapping[str, object],
    *,
    path: str,
    kind: str,
    lang: str,
    database: str | None = None,
) -> tuple[str, list[dict[str, object]]] | None:
    """Save a data-source path and return the assistant follow-up."""
    clean_path = str(path or "").strip()
    if not clean_path:
        return None
    clean_kind = kind if kind in {"prepared_path", "module_export", "raw_files"} else "prepared_path"
    prepared_invalid = False
    if clean_kind == "module_export":
        _copilot_set_module_export_path_in_chat(state, clean_path, database=database)
        status = (
            "module export folder recorded"
            if lang == "en" else
            "模块导出文件夹已记录"
        )
        next_sentence = (
            "Because this is already an EasyICU export, Agent setup can use it after you choose the cohort and modules."
            if lang == "en" else
            "这是已有 EasyICU 导出；选择队列和模块后，Agent 配置可以直接使用它。"
        )
    elif clean_kind == "raw_files":
        _copilot_set_raw_icu_path_in_chat(state, clean_path, database=database)
        status = (
            "raw ICU root recorded"
            if lang == "en" else
            "ICU 原始文件目录已记录"
        )
        next_sentence = (
            "This still needs validation/conversion before analysis; I will keep the conversion requirement visible in the study rail."
            if lang == "en" else
            "这仍需先验证/转换后才能分析；我会把转换需求保留在右侧进度里。"
        )
    else:
        _copilot_set_real_data_path_in_chat(state, clean_path, database=database)
        # D2: reflect the actual validation result instead of always saying
        # "pending validation". A folder that doesn't validate keeps the user on
        # the data step with a fix/convert path rather than silently advancing.
        prepared_valid = bool(state.get("path_validated"))
        if prepared_valid:
            status = (
                "prepared path validated and ready" if lang == "en"
                else "prepared 路径已验证、可用"
            )
            next_sentence = (
                "Next, choose the cohort scope in this same chat."
                if lang == "en" else
                "下一步继续在当前聊天里选择队列范围。"
            )
        else:
            prepared_invalid = True
            status = (
                "couldn't validate this folder as a converted EasyICU dataset"
                if lang == "en" else
                "无法把该文件夹验证为已转换的 EasyICU 数据集"
            )
            next_sentence = (
                "Open Classic Data Extraction to Validate / Convert & Setup, or paste a different converted path."
                if lang == "en" else
                "请用经典数据提取做 Validate / Convert & Setup,或换一个已转换的路径。"
            )
    state.pop("_copilot_data_source_choice", None)
    study = _ensure_copilot_study_state(state)
    study["step"] = "data" if prepared_invalid else "cohort"
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    label = _copilot_data_source_choice_label(clean_kind, lang)
    body = (
        f"Saved **{label}**: `{clean_path}`.\n\n{status}. {next_sentence}"
        if lang == "en" else
        f"已保存 **{label}**：`{clean_path}`。\n\n{status}。{next_sentence}"
    )
    actions = _copilot_guided_choice_actions(study, lang)
    if prepared_invalid:
        actions = [
            {
                "id": "workflow_study_extract",
                "kind": "workflow",
                "label": "Open classic extraction" if lang == "en" else "打开经典提取",
                "workflow": "study_extract",
            }
        ] + actions
    return body, actions


def _copilot_invalidate_downstream_extraction(
    state: MutableMapping[str, object],
    study: MutableMapping[str, object],
) -> None:
    """Drop stale extraction/review results when an upstream step changes.

    Editing the cohort, feature modules, or data path after an extraction must
    invalidate the loaded frame + review so the next run re-pulls — otherwise the
    chat would show review/agent results that no longer match the new inputs.
    """
    for key in (
        "loaded_concepts",
        "_extraction_done",
        "_review_workspace_ready",
        "patient_ids",
        "all_patient_count",
        "id_col",
        "loaded_data_origin",
    ):
        state.pop(key, None)
    study.pop("extracted", None)
    study.pop("reviewed", None)


def _copilot_submit_cohort_filter(
    state: MutableMapping[str, object],
    *,
    disease: str,
    age_min: object = None,
    age_max: object = None,
    los_min: object = None,
    first_icu: str = "yes",
    gender: str = "any",
    survival: str = "any",
    si_mode: object = None,
    lang: str,
) -> tuple[str, list[dict[str, object]]]:
    """Save classic Step 2 cohort filters from the Copilot page."""
    clean_disease = str(disease or "none").strip()
    if clean_disease != "none" and clean_disease not in _copilot_full_disease_options(lang):
        clean_disease = "none"
    clean_gender = str(gender or "any").strip()
    clean_survival = str(survival or "any").strip()
    clean_first_icu = str(first_icu or "any").strip()
    min_age = _copilot_parse_optional_int(age_min)
    max_age = _copilot_parse_optional_int(age_max)
    min_los = _copilot_parse_optional_int(los_min)

    # Sepsis suspected-infection mode drives the classic Sepsis-3 cohort via the
    # same session key the classic view reads (cohort_filters.py).
    clean_si_mode = str(si_mode or "").strip()
    if clean_si_mode and clean_si_mode in _copilot_sepsis_mode_options(lang):
        state["sepsis_si_mode"] = clean_si_mode

    cohort_filter = _copilot_default_cohort_filter()
    cohort_filter["age_min"] = min_age
    cohort_filter["age_max"] = max_age
    cohort_filter["los_min"] = min_los
    cohort_filter["disease_cohort"] = clean_disease
    cohort_filter["has_sepsis"] = True if clean_disease == "sepsis" else None
    cohort_filter["gender"] = clean_gender if clean_gender in {"M", "F"} else None
    if clean_survival == "survived":
        cohort_filter["survived"] = True
    elif clean_survival == "deceased":
        cohort_filter["survived"] = False
    if clean_first_icu == "yes":
        cohort_filter["first_icu_stay"] = True
    elif clean_first_icu == "no":
        cohort_filter["first_icu_stay"] = False

    filter_labels: list[str] = []
    if cohort_filter["first_icu_stay"] is True:
        filter_labels.append("first ICU stay")
    elif cohort_filter["first_icu_stay"] is False:
        filter_labels.append("readmissions only")
    if min_age is not None and max_age is not None:
        filter_labels.append(f"age {min_age}-{max_age}")
    elif min_age is not None:
        filter_labels.append(f"age >= {min_age}")
    elif max_age is not None:
        filter_labels.append(f"age <= {max_age}")
    if min_los is not None:
        filter_labels.append(f"ICU LOS >= {min_los}h")
    if clean_disease == "sepsis" and clean_si_mode and clean_si_mode != "auto":
        filter_labels.append(f"sepsis def: {clean_si_mode}")
    if clean_gender in {"M", "F"}:
        filter_labels.append(f"sex = {clean_gender}")
    if clean_survival == "survived":
        filter_labels.append("survived")
    elif clean_survival == "deceased":
        filter_labels.append("deceased")
    if clean_disease != "none":
        disease_label = _copilot_disease_label(clean_disease, lang)
        filter_labels.append(disease_label)

    state["cohort_filter"] = cohort_filter
    state["cohort_enabled"] = bool(filter_labels)
    state["filtered_patient_count"] = None
    _copilot_confirm_classic_step2(state)
    # Execute the SAME cohort filter the classic view runs, so the chat-path
    # cohort is byte-identical to the classic-path cohort (no-op on demo data).
    # Previously this point only recorded the filter and left the count unrun.
    _copilot_run_step(state, "cohort")
    study = _ensure_copilot_study_state(state)
    # D3: cohort changed → any prior extraction/review is stale.
    _copilot_invalidate_downstream_extraction(state, study)
    study["cohort_filters"] = filter_labels
    study["cohort_strategy"] = "filtered" if filter_labels else "eligible"
    study["cohort_configured"] = True
    study.pop("cohort_substep", None)
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    # D1: real-data cohort returned 0 patients — guide to loosen rather than
    # advancing into an empty study. (Only when a real filter actually ran;
    # eligible/demo leaves filtered_patient_count None and skips this.)
    matched = state.get("filtered_patient_count")
    if filter_labels and isinstance(matched, int) and matched == 0:
        study["step"] = "cohort"
        study["cohort_phase"] = "empty"
        body = (
            "These cohort filters matched **0 patients** in this dataset. "
            "Loosen them — widen age/LOS, drop the disease filter, or use the eligible cohort — then save again."
            if lang == "en" else
            "这些队列过滤条件在本数据集里匹配到 **0 位患者**。"
            "请放宽条件——放宽年龄/住院时长、去掉疾病过滤,或改用合格队列——再保存一次。"
        )
        return _copilot_reply(study, body, lang, include_status=False), [
            {
                "id": "workflow_study_loosen_filters",
                "kind": "workflow",
                "label": "Loosen filters" if lang == "en" else "放宽过滤条件",
                "workflow": "study_loosen_filters",
            },
            _copilot_prompt_action(
                "choice_eligible_cohort",
                "Use eligible cohort",
                "改用合格队列",
                "use the eligible cohort",
                "使用合格队列",
                lang,
            ),
        ]
    study["step"] = "concepts"
    summary = ", ".join(filter_labels) if filter_labels else ("eligible cohort" if lang == "en" else "合格队列")
    body = (
        f"Cohort filters saved in Copilot: **{summary}**. Next, choose feature modules; I will keep those selections synced to classic Step 3."
        if lang == "en" else
        f"队列筛选已在 Copilot 中保存：**{summary}**。下一步选择特征模块；我会同步到经典 Step 3。"
    )
    return body, _copilot_guided_choice_actions(study, lang)


def _copilot_submit_feature_modules(
    state: MutableMapping[str, object],
    *,
    module_keys: list[str],
    lang: str,
) -> tuple[str, list[dict[str, object]]] | None:
    """Save classic Step 3 feature-module selections from the Copilot page."""
    from easyicu.webapp.llm_chat import COPILOT_FEATURE_MODULE_PACKS  # lazy: avoid import cycle
    valid_keys = [key for key in module_keys if key in COPILOT_FEATURE_MODULE_PACKS]
    if not valid_keys:
        return None
    selected_concepts: list[str] = []
    for key in valid_keys:
        for concept in COPILOT_FEATURE_MODULE_PACKS[key]["concepts"]:
            if concept not in selected_concepts:
                selected_concepts.append(str(concept))
    module_labels = [_copilot_feature_pack_label(key, lang) for key in valid_keys]
    state["selected_concepts"] = selected_concepts
    _copilot_confirm_classic_step3(state)
    study = _ensure_copilot_study_state(state)
    # D3: feature selection changed → any prior extraction/review is stale.
    _copilot_invalidate_downstream_extraction(state, study)
    study["selected_concepts"] = selected_concepts
    study["modules"] = module_labels
    study["concepts_configured"] = True
    study["step"] = "extract"
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    concept_labels = _copilot_concept_label_list(study, limit=10)
    body = (
        "Feature modules saved in Copilot: "
        f"**{', '.join(module_labels)}**.\n\n"
        f"Selected concepts synced to classic Step 3: `{', '.join(concept_labels)}`. "
        "Next, I can assemble the extraction plan in this chat before any Agent run."
        if lang == "en" else
        "特征模块已在 Copilot 中保存："
        f"**{'、'.join(module_labels)}**。\n\n"
        f"已同步到经典 Step 3 的概念：`{'、'.join(concept_labels)}`。"
        "下一步我会在当前聊天中组装提取计划，再进入 Agent。"
    )
    return body, _copilot_guided_choice_actions(study, lang)


def _copilot_apply_entities(study: MutableMapping[str, object], text: str) -> list[str]:
    text_l = (text or "").lower()
    found: list[str] = []
    existing_concepts = [
        str(item)
        for item in list(study.get("selected_concepts") or [])
        if str(item).strip()
    ]
    parsed_concepts = _copilot_concepts_from_text(text)
    for concept in parsed_concepts:
        if concept not in existing_concepts:
            existing_concepts.append(concept)
            found.append(concept)
    if parsed_concepts:
        study["selected_concepts"] = existing_concepts
        study["modules"] = _copilot_modules_for_concepts(existing_concepts)
        study["concepts_configured"] = True
    exposure_aliases = [
        (r"\blactate\b|乳酸", "lactate"),
        (r"\bsofa\b|sofa-?2", "SOFA"),
        (r"\bmap\b|mean arterial", "MAP"),
        (r"creatinine|肌酐", "creatinine"),
        (r"heart rate|心率", "heart rate"),
        (r"\bwbc\b|white cell|白细胞", "WBC"),
    ]
    for pattern, label in exposure_aliases:
        if re.search(pattern, text_l):
            study["exposure"] = label
            found.append(label)
            break
    window_match = re.search(r"(?:first\s*)?(\d{1,3})\s*(?:h\b|hr|hour|小时)", text_l)
    if window_match:
        study["window"] = f"first {window_match.group(1)}h"
        found.append(str(study["window"]))
    if re.search(r"28[\s-]*day|28\s*天", text_l):
        study["outcome"] = "28-day mortality"
        found.append("28-day mortality")
    elif re.search(r"icu\s+mortality|icu\s+death|icu\s*死亡", text_l):
        study["outcome"] = "ICU mortality"
        found.append("ICU mortality")
    elif re.search(r"in-?hospital|院内", text_l):
        study["outcome"] = "In-hospital mortality"
        found.append("in-hospital mortality")
    elif re.search(r"\baki\b|\brrt\b|renal|kidney|creatinine|urine output|肾脏替代|急性肾|肾损伤|肾损害|肾功能|肾衰|肌酐|尿量", text_l):
        study["outcome"] = "AKI / RRT"
        found.append("AKI / RRT")
    elif re.search(r"length\s+of\s+stay|\blos\b|住院时长|住院时间", text_l):
        study["outcome"] = "ICU length of stay"
        found.append("ICU length of stay")
    patient_n = _copilot_extract_patient_count(text)
    if patient_n is not None:
        study["patient_n"] = patient_n
        study["cohort_configured"] = True
        found.append(f"{patient_n} stays")
    if any(term in text_l for term in ("sepsis-3", "sepsis 3", "sepsis", "脓毒症")):
        filters = list(study.get("cohort_filters") or [])
        if "sepsis-3" not in filters:
            filters.append("sepsis-3")
        study["cohort_filters"] = filters
        if "sepsis-3" not in found:
            found.append("sepsis-3")
    if any(term in text_l for term in ("first icu", "首次 icu", "first stay", "首次住院")):
        filters = list(study.get("cohort_filters") or [])
        if "first ICU stay" not in filters:
            filters.append("first ICU stay")
        study["cohort_filters"] = filters
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    return found


def _copilot_advance_step(study: MutableMapping[str, object]) -> str:
    current = str(study.get("step") or "question")
    # Depth-gated advance: never auto-advance past the study's depth goal
    # (extract / review / full). The finish-line UI offers "take it further".
    next_step = _copilot_engine.next_step_capped(study.get("depth"), current)
    study["step"] = next_step
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    return next_step


def _copilot_submit_extract(
    state: MutableMapping[str, object],
    lang: str,
) -> tuple[str, list[dict[str, object]]]:
    """Run the extract step deterministically (no LLM route) via the classic engine.

    Mirrors `_copilot_submit_cohort_filter`: a fixed step button executes the
    SAME `load_data_for_preview` the classic view uses, then summarises results
    in chat and respects the depth goal (stops at the extract finish line when
    depth == extract).
    """
    is_en = lang == "en"
    study = _ensure_copilot_study_state(state)
    study["step"] = "extract"
    # The classic preview loader reads state["selected_concepts"]/data_path; make
    # sure the study's concept selection is synced into session state first, or
    # the loader returns nothing ("0 tables"). Keep the real-data binding intact.
    if str(study.get("data_mode") or "") == "real" or str(state.get("data_path") or "").strip():
        state["entry_mode"] = "real"
        state["use_mock_data"] = False
        if state.get("database") in {None, "", "mock"}:
            state["database"] = _copilot_normalize_database(state.get("database") or "miiv")
        # Ensure the path is validated/confirmed like classic Step 1 before load.
        if str(state.get("data_path") or "").strip() and not state.get("path_validated"):
            _copilot_validate_real_data_path(state)
    _apply_copilot_study_to_workspace(state)
    result = _copilot_run_step(state, "extract")
    status = str(result.get("status") or "")

    if status == "no_real_data":
        body = (
            "I don't have a real data source bound yet, so there's nothing to extract. "
            "Paste a prepared path with `set data path /path/to/prepared_data`, then ask me to prepare the extraction plan again."
            if is_en else
            "目前还没有绑定真实数据源,无法提取。请先用 `set data path /路径/到/prepared_data` 设置 prepared 路径,再让我准备提取计划。"
        )
        return _copilot_reply(study, body, lang, include_status=False), [
            _copilot_prompt_action(
                "choice_set_data_path",
                "Set data path",
                "设置数据路径",
                "what real data path should I use?",
                "真实数据路径应该填什么？",
                lang,
            )
        ]

    if status == "error":
        err = str(result.get("error") or "")
        body = (
            f"Extraction hit an error: `{err}`. The path may not be a prepared/converted EasyICU dataset. "
            "Open the classic Data Extraction view for the full validate/convert flow, or set a different path."
            if is_en else
            f"提取出错:`{err}`。该路径可能不是已转换的 prepared EasyICU 数据集。可以打开经典数据提取视图走完整的验证/转换流程,或换一个路径。"
        )
        return _copilot_reply(study, body, lang, include_status=False), [
            {
                "id": "workflow_study_extract",
                "kind": "workflow",
                "label": "Open classic extraction" if is_en else "打开经典提取",
                "workflow": "study_extract",
            }
        ]

    # status == "ok"
    loaded = state.get("loaded_concepts")
    n_tables = len(loaded) if isinstance(loaded, Mapping) else 0
    if n_tables == 0:
        # The engine returns "ok" even when the loader yielded nothing; don't
        # pretend success. Surface it honestly with a path/validation hint.
        validated = bool(state.get("path_validated"))
        hint = (
            "the path validated but no concept tables came back — check the database selection and selected modules"
            if validated else
            "the data path isn't validated yet (it may need conversion); open the classic Data Extraction view to validate/convert"
        ) if is_en else (
            "路径已验证但没有载入任何概念表——请检查数据库选择和所选模块"
            if validated else
            "数据路径尚未验证(可能需要转换);打开经典数据提取视图做验证/转换"
        )
        body = (
            f"Extraction returned 0 concept tables. Likely cause: {hint}."
            if is_en else
            f"提取返回 0 张概念表。可能原因:{hint}。"
        )
        return _copilot_reply(study, body, lang, include_status=False), [
            {
                "id": "workflow_study_extract",
                "kind": "workflow",
                "label": "Open classic extraction" if is_en else "打开经典提取",
                "workflow": "study_extract",
            }
        ]
    cohort_n = state.get("filtered_patient_count")
    if not isinstance(cohort_n, int):
        stats = state.get("_cohort_stats")
        cohort_n = stats.get("total_after") if isinstance(stats, Mapping) else None
    study["extracted"] = True
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    next_step = _copilot_advance_step(study)

    cohort_txt = (
        (f" for ~{cohort_n} stays" if is_en else f"(约 {cohort_n} 例 stay)")
        if isinstance(cohort_n, int) else ""
    )
    base = (
        f"Extraction preview is ready{cohort_txt}: {n_tables} concept tables loaded through the same classic pipeline."
        if is_en else
        f"提取预览已就绪{cohort_txt}:已通过与经典一致的管线载入 {n_tables} 张概念表。"
    )
    if next_step == "extract":
        # Depth goal reached — stop at the finish line.
        tail = (
            " This is your finish line for the Extract depth. Use \"Take it further\" on the right to continue to Review, or open the classic view for detailed audit."
            if is_en else
            "这是 Extract 档的终点线。可点右侧的「继续深入」进入审阅,或打开经典视图做细节审阅。"
        )
    else:
        tail = (
            " Next: visual review of the extracted cohort."
            if is_en else
            "下一步:对提取队列做可视化审阅。"
        )
    body = _copilot_reply(study, base + tail, lang, include_status=False)
    return body, _copilot_guided_choice_actions(study, lang)


def _copilot_submit_review(
    state: MutableMapping[str, object],
    lang: str,
) -> tuple[str, list[dict[str, object]]]:
    """Run the review step deterministically (no LLM route) via the classic engine.

    Mirrors `_copilot_submit_extract`: marks the review workspace ready, surfaces
    a metric summary from the SAME `quick_visualization_page` helper, links to the
    embedded classic Patient Review, and respects the depth finish line.
    """
    is_en = lang == "en"
    study = _ensure_copilot_study_state(state)
    study["step"] = "review"

    loaded = state.get("loaded_concepts")
    n_tables = len(loaded) if isinstance(loaded, Mapping) else 0
    if n_tables == 0:
        # Nothing extracted yet — guide back to the extract step rather than
        # opening an empty review.
        body = (
            "There's no extracted cohort to review yet. Run **Prepare extraction plan** first, "
            "then I'll open the visual review here."
            if is_en else
            "目前还没有可审阅的提取队列。请先运行 **准备提取计划**，我再在这里打开可视化审阅。"
        )
        return _copilot_reply(study, body, lang, include_status=False), [
            _copilot_prompt_action(
                "choice_prepare_extraction",
                "Prepare extraction plan",
                "准备提取计划",
                "Prepare the extraction plan in chat.",
                "在聊天里准备提取计划。",
                lang,
            )
        ]

    # Mark the review workspace ready through the classic engine + enable the
    # classic preview so the embedded Patient Review renders the same data.
    _copilot_run_step(state, "review")
    state["_preview_requested"] = True

    summary: Mapping[str, object] = {}
    try:
        from easyicu.webapp.quick_visualization_page import _quick_viz_workspace_summary
        summary = _quick_viz_workspace_summary(dict(state), lang) or {}
    except Exception:  # pragma: no cover - defensive
        summary = {}
    concept_count = int(summary.get("concept_count") or n_tables)
    patient_count = summary.get("loaded_patient_count") or summary.get("all_patient_count")
    cohort_n = state.get("filtered_patient_count")
    if not isinstance(cohort_n, int):
        stats = state.get("_cohort_stats")
        cohort_n = stats.get("total_after") if isinstance(stats, Mapping) else None

    study["reviewed"] = True
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    next_step = _copilot_advance_step(study)

    bits = []
    bits.append(
        (f"{concept_count} concept tables" if is_en else f"{concept_count} 张概念表")
    )
    if isinstance(patient_count, int) and patient_count:
        bits.append((f"{patient_count} patients in preview" if is_en else f"{patient_count} 位患者预览"))
    elif isinstance(cohort_n, int):
        bits.append((f"~{cohort_n} cohort stays" if is_en else f"约 {cohort_n} 例队列 stay"))
    summary_txt = ", ".join(bits)

    base = (
        f"Visual review is ready: {summary_txt}. The Patient Review workspace below is bound to the same extracted frame."
        if is_en else
        f"可视化审阅已就绪:{summary_txt}。下方的患者审阅工作区与提取出的同一数据帧绑定。"
    )
    if next_step == "review":
        tail = (
            " This is your finish line for the Review depth. Use \"Take it further\" on the right to continue to the agent analysis."
            if is_en else
            "这是 Review 档的终点线。可点右侧的「继续深入」进入 agent 分析。"
        )
    else:
        tail = (
            " Next: the evidence-bound agent analysis run."
            if is_en else
            "下一步:证据绑定的 agent 分析运行。"
        )
    actions = [
        {
            "id": "workflow_study_review",
            "kind": "workflow",
            "label": "Open Patient Review" if is_en else "打开患者审阅",
            "workflow": "study_review",
        }
    ]
    body = _copilot_reply(study, base + tail, lang, include_status=False)
    return body, actions


def _copilot_run_step(
    state: MutableMapping[str, object],
    step_id: str,
    **kwargs: object,
) -> dict[str, object]:
    """Route one Copilot step to the SAME classic engine the classic views use.

    Thin wrapper over ``copilot_engine.run_copilot_step``: the live
    ``st.session_state`` is both the state read and written, so a cohort filtered
    in chat lands in the exact keys (``cohort_filter`` / ``_cohort_stats`` /
    ``_cohort_filtered_ids``) the classic Data Extraction view uses. Guarded so a
    data-layer error degrades to a no-op instead of breaking the chat surface.
    """
    study = _ensure_copilot_study_state(state)
    app_context = state.get("_app_context")
    try:
        return _copilot_engine.run_copilot_step(
            step_id, study, state, app_context=app_context, **kwargs
        )
    except Exception as exc:  # pragma: no cover - defensive UI guard
        return {"step": step_id, "status": "error", "error": str(exc)}
