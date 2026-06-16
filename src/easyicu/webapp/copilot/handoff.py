"""Research Copilot — study-state -> research-agent handoff + workspace seeding.

Extracted from `llm_chat.py` (Phase-6 split, 8th batch). Translates a copilot
study (cohort + concepts + framing) into a research-agent run: seeds the agent
workspace from the study, wires the real prepared-data source / patient-id
handoff, prepares the AI- and idea-driven handoffs, frames the study question,
and seeds the cross-database demo workspace. Pure helpers operating on `state` /
`study` mappings — **no Streamlit `st` access** — so the test suite's
`monkeypatch.setattr(llm_chat, "st", ...)` contract does not apply.

Cross-module leaf imports (none import this module, so no cycle):
`clear_agent_continuation_state` (session_state), `_research_agent_source_label`
(presentation), `_ensure_copilot_study_state` (sessions). The one
`COPILOT_BRANCH_CONFIG` constant still in `llm_chat.py` is lazy-imported inside
the using functions. `llm_chat.py` re-imports every name below.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping

from easyicu.webapp.copilot.presentation import _research_agent_source_label
from easyicu.webapp.copilot.sessions import _ensure_copilot_study_state
from easyicu.webapp.session_state import clear_agent_continuation_state


def _copilot_template_for_study(study: MutableMapping[str, object]) -> str:
    """Map the conversational branch into a Research Agent template key."""
    branch = str(study.get("branch") or "predict")
    question = str(study.get("question") or "").lower()
    if branch == "crossdb":
        return "validation"
    if branch == "quality":
        return "data_quality"
    if any(term in question for term in ("association", "associated", "risk factor", "odds ratio", "or ")):
        return "association"
    if any(term in question for term in ("survival", "time-to-event", "cox", "28-day")):
        return "survival"
    return "prediction"


def _copilot_outcome_for_study(study: MutableMapping[str, object]) -> str:
    outcome = str(study.get("outcome") or "").lower()
    if "icu" in outcome:
        return "death"
    if "28" in outcome:
        return "death"
    if "mortality" in outcome or "death" in outcome:
        return "death"
    return "death"


def _copilot_frame_question(study: MutableMapping[str, object], lang: str) -> str:
    from easyicu.webapp.llm_chat import COPILOT_BRANCH_CONFIG  # lazy
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])
    if branch == "predict":
        window = str(study.get("window") or ("first 24h" if lang == "en" else "前 24 小时"))
        raw_outcome = str(study.get("outcome") or "").strip()
        question_kind = str(study.get("question_kind") or "outcome_model")
        exposure = str(study.get("exposure") or "").strip()
        if question_kind == "treatment_exposure" and exposure:
            if lang == "en":
                outcome = raw_outcome or "a prespecified ICU outcome"
                outcome_phrase = outcome if "ICU" in outcome else outcome.lower()
                return (
                    f"Is {window} {exposure} associated with {outcome_phrase} in the selected ICU cohort, "
                    "after cohort, severity, and data-availability checks?"
                )
            outcome = raw_outcome or "预设 ICU 结局"
            return (
                f"{window}{exposure}是否与用户选择的 ICU 队列中的{outcome}相关，"
                "并通过队列、严重程度和数据可用性检查？"
            )
        if lang == "en":
            outcome = raw_outcome or "a prespecified ICU outcome"
            outcome_phrase = outcome if "ICU" in outcome else outcome.lower()
            return (
                f"Do {window} bedside features predict {outcome_phrase} in the selected ICU cohort, "
                "and which added feature modules improve the model?"
            )
        outcome = raw_outcome or "预设 ICU 结局"
        return (
            f"{window}床旁特征能否在用户选择的 ICU 队列中预测{outcome}，哪些新增特征模块能改善模型？"
        )
    return str(config["question_en"] if lang == "en" else config["question_zh"])


def _seed_demo_context_from_chat(state: MutableMapping[str, object]) -> None:
    """Put the classic workspace into a safe demo-ready state."""
    study = state.get("_copilot_guided_study") if isinstance(state.get("_copilot_guided_study"), dict) else {}
    current_mock_params = state.get("mock_params") if isinstance(state.get("mock_params"), dict) else {}
    patient_n = int(
        (study or {}).get("patient_n")
        or current_mock_params.get("n_patients")
        or state.get("demo_mode_patients")
        or 10
    )
    demo_hours = int(
        (study or {}).get("hours")
        or current_mock_params.get("hours")
        or state.get("demo_mode_hours")
        or 24
    )
    state["entry_mode"] = "demo"
    state["use_mock_data"] = True
    state["database"] = "mock"
    state["mock_params"] = {
        "n_patients": patient_n,
        "hours": demo_hours,
        "demo_profile": "lite",
    }
    state["_eu_demo_widget_params_pending"] = {
        "n_patients": int(state["mock_params"]["n_patients"]),
        "hours": int(state["mock_params"]["hours"]),
    }
    for key in ("step1_confirmed", "step2_confirmed", "step3_confirmed", "export_completed"):
        state[key] = False
    state["trigger_export"] = False
    state["_exporting_in_progress"] = False


def _copilot_selected_concepts_for_study(state: MutableMapping[str, object]) -> list[str]:
    from easyicu.webapp.llm_chat import COPILOT_BRANCH_CONFIG  # lazy
    study = _ensure_copilot_study_state(state)
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])
    concepts = [
        str(item)
        for item in list(study.get("selected_concepts") or [])
        if str(item).strip()
    ]
    if not concepts:
        concepts = list(config.get("selected_concepts") or [])
    return concepts or ["hr", "map", "temp", "spo2", "sofa2"]


def _copilot_patient_ids_for_handoff(state: Mapping[str, object]) -> list[object] | None:
    raw = state.get("patient_ids")
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, tuple):
        return list(raw)
    if isinstance(raw, set):
        return list(raw)
    tolist = getattr(raw, "tolist", None)
    if callable(tolist):
        try:
            values = tolist()
        except Exception:
            return None
        return list(values) if isinstance(values, list) else None
    return None


def _latest_module_export_dir_for_handoff(state: MutableMapping[str, object]) -> str:
    """Prefer the Research Agent's own module-export folder resolver."""
    try:
        from easyicu.webapp.research_agent import _module_folder_manual_handoff_dir

        resolved = str(_module_folder_manual_handoff_dir(state) or "").strip()
        if resolved:
            return resolved
    except Exception:
        pass
    for key in ("last_export_dir", "export_path"):
        raw = str(state.get(key) or "").strip()
        if raw:
            return raw
    return ""


def _clear_copilot_agent_cohort_handoff(state: MutableMapping[str, object]) -> None:
    """Drop stale Copilot cohort state before switching source modes."""
    for key in (
        "research_agent_inbound_cohort",
        "research_agent_inbound_cohort_label",
        "research_agent_inbound_signature",
    ):
        state.pop(key, None)
    state.pop("_eu_ra_force_setup_from_handoff", None)
    state.pop("_research_agent_previous_cohort_source", None)


def _copilot_study_should_use_real_source(
    study: Mapping[str, object],
    state: Mapping[str, object],
) -> bool:
    data_mode = str(study.get("data_mode") or "").strip().lower()
    if data_mode == "real":
        return True
    if str(state.get("entry_mode") or "").strip().lower() == "real":
        return True
    if state.get("use_mock_data") is False:
        return True
    return False


def _seed_research_agent_real_source_from_copilot(
    state: MutableMapping[str, object],
    *,
    lang: str,
) -> bool:
    """Route Copilot handoff through real session data or module exports."""
    state["entry_mode"] = "real"
    state["use_mock_data"] = False
    if state.get("database") == "mock":
        state["database"] = "miiv"

    export_dir = _latest_module_export_dir_for_handoff(state)
    loaded_concepts = state.get("loaded_concepts")
    if isinstance(loaded_concepts, Mapping) and loaded_concepts:
        try:
            from easyicu.webapp.research_agent import _stay_level_from_loaded_concepts

            id_col = str(state.get("id_col") or "stay_id")
            patient_ids = _copilot_patient_ids_for_handoff(state)
            cohort = _stay_level_from_loaded_concepts(
                dict(loaded_concepts),
                id_col=id_col,
                patient_ids=patient_ids,
            )
        except Exception:
            cohort = None
        if cohort is not None and not getattr(cohort, "empty", True):
            rows = len(cohort)
            concepts = len(loaded_concepts)
            state["research_agent_inbound_cohort"] = cohort
            state["research_agent_inbound_cohort_label"] = (
                f"Research Copilot loaded export · {rows} stays · {concepts} concepts"
                if lang == "en" else
                f"Research Copilot 已加载导出 · {rows} 例 stay · {concepts} 个概念"
            )
            state["research_agent_inbound_signature"] = (
                "research_copilot_loaded_concepts",
                tuple(sorted(str(key) for key in loaded_concepts.keys())),
                rows,
                str(state.get("id_col") or "stay_id"),
                tuple(str(item) for item in (patient_ids or [])),
            )
            state["research_agent_cohort_source"] = _research_agent_source_label("handoff", lang)
            state["_research_agent_previous_cohort_source"] = None
            state["_eu_ra_force_setup_from_handoff"] = True
            state.pop("_eu_ra_focus_no_data", None)
            if export_dir:
                state["research_agent_module_dir_text"] = export_dir
                state["_eu_ra_module_pick_force_manual"] = True
                state["_eu_ra_apply_export_file_selection"] = True
                state.pop("research_agent_module_dir_pick", None)
            return True

    _clear_copilot_agent_cohort_handoff(state)
    if export_dir:
        state["research_agent_cohort_source"] = _research_agent_source_label("module", lang)
        state["research_agent_module_dir_text"] = export_dir
        state["_eu_ra_focus_module_folder"] = True
        state["_eu_ra_module_pick_force_manual"] = True
        state["_eu_ra_apply_export_file_selection"] = True
        state.pop("research_agent_module_dir_pick", None)
        state.pop("_eu_ra_focus_no_data", None)
        return True

    state["research_agent_cohort_source"] = _research_agent_source_label("no_data", lang)
    state["_eu_ra_focus_no_data"] = True
    state.pop("_eu_ra_focus_module_folder", None)
    return False


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, Mapping):
        raw_items = list(value.keys())
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = []
    return [str(item) for item in raw_items if str(item).strip()]


def _research_agent_handoff_packet_source(
    state: Mapping[str, object],
    *,
    lang: str,
) -> tuple[str, str, int | None, str]:
    """Summarize the active Agent source as a stable handoff packet row."""
    inbound_label = str(state.get("research_agent_inbound_cohort_label") or "").strip()
    inbound = state.get("research_agent_inbound_cohort")
    rows: int | None = None
    if inbound is not None:
        try:
            rows = int(len(inbound))  # type: ignore[arg-type]
        except Exception:
            rows = None
    export_dir = str(state.get("research_agent_module_dir_text") or "").strip()
    if inbound_label:
        return "session_cohort", inbound_label, rows, export_dir
    if export_dir and state.get("_eu_ra_focus_module_folder"):
        label = "Module export folder" if lang == "en" else "模块导出目录"
        return "module_export", label, None, export_dir
    if state.get("_eu_ra_focus_no_data"):
        label = "No cohort loaded yet" if lang == "en" else "尚未加载队列"
        return "no_data", label, None, export_dir
    if str(state.get("entry_mode") or "").strip().lower() == "demo":
        label = "Demo cohort prepared by Copilot" if lang == "en" else "Copilot 准备的演示队列"
        return "demo_cohort", label, rows, export_dir
    label = str(state.get("research_agent_cohort_source") or "").strip()
    return "configured_source", label, rows, export_dir


def _store_research_agent_copilot_handoff_packet(
    state: MutableMapping[str, object],
    *,
    lang: str,
    source_kind: str | None = None,
) -> None:
    """Persist a compact handoff packet for the Agent setup UI."""
    context = state.get("research_agent_copilot_context")
    context_map: Mapping[str, object] = context if isinstance(context, Mapping) else {}
    source_key, source_label, rows, export_dir = _research_agent_handoff_packet_source(state, lang=lang)
    if source_kind:
        source_key = source_kind
    selected_raw = context_map.get("selected_concepts") or state.get("selected_concepts") or []
    selected = _string_list(selected_raw)
    filters_raw = context_map.get("cohort_filters") or []
    cohort_filters = _string_list(filters_raw)
    loaded_concepts = state.get("loaded_concepts")
    loaded_concept_count = len(loaded_concepts) if isinstance(loaded_concepts, Mapping) else 0
    question = str(
        state.get("research_agent_question")
        or context_map.get("question")
        or state.get("_copilot_last_question")
        or ""
    ).strip()
    if not question:
        state.pop("research_agent_copilot_handoff_packet", None)
        return

    state["research_agent_copilot_handoff_packet"] = {
        "source": "Research Copilot",
        "source_kind": source_key,
        "source_label": source_label,
        "question": question[:1200],
        "branch": str(context_map.get("branch") or "predict"),
        "data_mode": str(context_map.get("data_mode") or state.get("entry_mode") or "demo"),
        "patient_n": _int_or_none(context_map.get("patient_n")),
        "window": str(context_map.get("window") or ""),
        "outcome": str(context_map.get("outcome") or state.get("research_agent_target_outcome") or ""),
        "exposure": str(context_map.get("exposure") or ""),
        "selected_concepts": selected[:12],
        "selected_count": len(selected),
        "concept_count": loaded_concept_count or len(selected),
        "cohort_filters": cohort_filters[:8],
        "template_key": str(context_map.get("template_key") or state.get("research_agent_template_current") or ""),
        "rows": rows,
        "export_dir": export_dir,
        "preflight_status": "needs_review",
        "next_step": (
            "Review setup, confirm preflight, then run Agent"
            if lang == "en" else
            "复核 setup，确认 preflight 后再运行 Agent"
        ),
    }


def _apply_copilot_study_to_workspace(state: MutableMapping[str, object]) -> None:
    study = _ensure_copilot_study_state(state)
    question = str(study.get("question") or _copilot_frame_question(study, state.get("language", "en"))).strip()
    if question:
        study["question"] = question
        state["_copilot_last_question"] = question
    state["selected_concepts"] = _copilot_selected_concepts_for_study(state)
    state["_preview_n"] = int(study.get("patient_n") or state.get("demo_mode_patients") or 10)
    state["_copilot_guided_study"] = study


def _seed_research_agent_from_copilot_study(state: MutableMapping[str, object]) -> bool:
    """Bind the current Copilot study into Research Agent setup context."""
    raw_study = state.get("_copilot_guided_study")
    if not isinstance(raw_study, dict):
        return False
    lang = str(state.get("language") or "en")
    is_en = lang == "en"
    had_agent_question = bool(str(state.get("research_agent_question") or "").strip())
    study = _ensure_copilot_study_state(state)
    _apply_copilot_study_to_workspace(state)
    question = str(study.get("question") or _copilot_frame_question(study, lang)).strip()
    if question and not str(state.get("research_agent_question") or "").strip():
        state["research_agent_question"] = question
        state["_research_agent_question_handoff_notice"] = True
    elif question:
        state["_copilot_last_question"] = question
    template_key = _copilot_template_for_study(study)
    state["research_agent_template_current"] = template_key
    state["research_agent_example_key"] = template_key
    state["research_agent_example_active"] = {
        "prediction": "Prediction model",
        "association": "Association",
        "survival": "Survival / time-to-event",
        "validation": "External validation / score benchmarking",
        "data_quality": "Data-quality / missingness / harmonization audit",
    }.get(template_key, "Prediction model")
    state["research_agent_target_outcome"] = _copilot_outcome_for_study(study)
    state["research_agent_workflow_mode"] = "analysis_run"
    state.pop("research_agent_workflow_mode_pick", None)
    state["research_agent_copilot_context"] = {
        "source": "research_copilot",
        "branch": str(study.get("branch") or "predict"),
        "question": question,
        "data_mode": str(study.get("data_mode") or "demo"),
        "patient_n": int(study.get("patient_n") or 10),
        "window": str(study.get("window") or "first 24h"),
        "outcome": str(study.get("outcome") or "In-hospital mortality"),
        "exposure": str(study.get("exposure") or "lactate"),
        "selected_concepts": list(state.get("selected_concepts") or []),
        "cohort_filters": list(study.get("cohort_filters") or []),
        "template_key": template_key,
    }
    if _copilot_study_should_use_real_source(study, state):
        _seed_research_agent_real_source_from_copilot(state, lang=lang)
    elif str(study.get("data_mode") or "demo") == "demo":
        try:
            from easyicu.webapp.research_agent import _build_synthetic_cohort

            requested_n = int(study.get("patient_n") or 10)
            cohort_n = max(5, min(800, requested_n))
            state["research_agent_inbound_cohort"] = _build_synthetic_cohort(n=cohort_n, seed=7)
            state["research_agent_inbound_cohort_label"] = (
                f"Research Copilot demo cohort · {cohort_n} stays"
                if is_en else
                f"Research Copilot 演示队列 · {cohort_n} 例 stay"
            )
            state["research_agent_cohort_source"] = "Session handoff" if is_en else "会话交接"
            state["_eu_ra_force_setup_from_handoff"] = True
            state.pop("_eu_ra_focus_module_folder", None)
            state.pop("_eu_ra_focus_no_data", None)
            state.pop("_eu_ra_module_pick_force_manual", None)
            state.pop("_eu_ra_apply_export_file_selection", None)
        except Exception:
            state.pop("research_agent_inbound_cohort", None)
            state.pop("research_agent_inbound_cohort_label", None)
    state.pop("research_agent_preflight_confirmed", None)
    state.pop("research_agent_preflight_ack", None)
    state.pop("research_agent_preflight_signature", None)
    _store_research_agent_copilot_handoff_packet(state, lang=lang)
    return bool(str(state.get("research_agent_question") or "").strip()) and not had_agent_question


def _seed_crossdb_demo_workspace(
    state: MutableMapping[str, object],
    *,
    concepts: list[str] | None = None,
) -> None:
    """Seed the classic Cross-DB demo benchmark with explicit or default concepts."""
    try:
        from easyicu.webapp.demo_data import (
            COHORT_DEMO_MULTIDB_DATABASES,
            COHORT_DEMO_MULTIDB_CONCEPTS,
            COHORT_DEMO_MULTIDB_RECORDS_PER_FEATURE,
            _generate_mock_multidb_data,
        )

        selected_concepts = concepts or list(COHORT_DEMO_MULTIDB_CONCEPTS)
        state["multidb_data"] = _generate_mock_multidb_data(
            str(state.get("language") or "en"),
            database_keys=COHORT_DEMO_MULTIDB_DATABASES,
            concepts=selected_concepts,
            records_per_feature=COHORT_DEMO_MULTIDB_RECORDS_PER_FEATURE,
        )
    except Exception:
        state.pop("multidb_data", None)
        selected_concepts = concepts or ["hr", "sbp", "map", "temp", "spo2", "lact"]
    state["entry_mode"] = "demo"
    state["use_mock_data"] = True
    state["database"] = "mock"
    state["multidb_concepts"] = selected_concepts
    state["multidb_is_demo"] = True
    state["_eu_crossdb_distribution_open"] = False


def _seed_copilot_crossdb_demo_workspace(state: MutableMapping[str, object]) -> None:
    """Seed Cross-DB demo data using the guided study's selected concepts."""
    _seed_crossdb_demo_workspace(state, concepts=_copilot_selected_concepts_for_study(state))


def _latest_ai_handoff_question(state: MutableMapping[str, object]) -> str:
    """Return the most recent assistant-side user request worth seeding."""
    pending = str(state.get("_ai_pending_question") or "").strip()
    if pending:
        return pending[:1200]

    study = state.get("_copilot_guided_study")
    if isinstance(study, dict):
        question = str(study.get("question") or "").strip()
        if question:
            return question[:1200]

    copilot_question = str(state.get("_copilot_last_question") or "").strip()
    if copilot_question:
        return copilot_question[:1200]

    messages = state.get("llm_messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").lower() != "user":
                continue
            content = str(message.get("content") or "").strip()
            if content:
                return content[:1200]
    return ""


def _prepare_research_agent_handoff_from_ai(state: MutableMapping[str, object]) -> bool:
    """Route Research Copilot to Agent setup and seed the question when safe."""
    clear_agent_continuation_state(state)
    seeded = _seed_research_agent_from_copilot_study(state)
    if not str(state.get("research_agent_question") or "").strip():
        handoff_question = _latest_ai_handoff_question(state)
        if handoff_question:
            state["research_agent_question"] = handoff_question
            state["_research_agent_question_handoff_notice"] = True
            seeded = True
    if seeded and not isinstance(state.get("research_agent_copilot_handoff_packet"), Mapping):
        _store_research_agent_copilot_handoff_packet(
            state,
            lang=str(state.get("language") or "en"),
            source_kind="copilot_prompt",
        )

    state["_active_main_page"] = "research_agent"
    state["_ra_view"] = "setup"
    state["_scroll_to_top"] = True
    if str(state.get("research_agent_question") or "").strip():
        state["_eu_ra_question_handoff_setup"] = True
    state["_inline_ai_panel_open"] = False
    state["_floating_ai_open"] = False
    state["_sidebar_ai_open"] = False
    state.pop("_ai_pending_question", None)
    return seeded


def _default_research_agent_idea_prompt(lang: str) -> str:
    if lang == "en":
        return (
            "Turn a review or editorial excerpt into an EasyICU idea exploration task: "
            "freeze the source, extract candidate hypotheses, map them to the current cohort, "
            "run outcome-blind feasibility, write the preregistration registry, and stop at the human gate."
        )
    return (
        "请把综述或 editorial 摘录转成 EasyICU idea 探索任务：冻结来源、提取候选假设、"
        "映射到当前队列、运行 outcome-blind 可行性检查、写入预注册 registry，并停在人工关口。"
    )


def _prepare_research_agent_idea_handoff_from_ai(state: MutableMapping[str, object]) -> bool:
    """Route Copilot to the Agent idea-mining dry-run without starting execution."""
    lang = str(state.get("language", "en"))
    clear_agent_continuation_state(state)
    handoff_question = _latest_ai_handoff_question(state) or _default_research_agent_idea_prompt(lang)
    seeded = False
    if not str(state.get("research_agent_question") or "").strip() and handoff_question:
        state["research_agent_question"] = handoff_question[:1200]
        state["_research_agent_question_handoff_notice"] = True
        seeded = True

    state["research_agent_example_active"] = (
        "Idea exploration / literature discovery"
        if lang == "en" else
        "Idea 探索 / 文献发现"
    )
    state["research_agent_example_key"] = "idea_exploration"
    state["research_agent_template_current"] = "idea_exploration"
    state["research_agent_workflow_mode"] = "idea_exploration"
    state.pop("research_agent_workflow_mode_pick", None)
    state["research_agent_preflight_confirmed"] = False
    state.pop("research_agent_preflight_signature", None)
    state.pop("research_agent_last_idea_result", None)
    state.pop("research_agent_last_idea_summary", None)
    state.pop("_research_agent_pending_idea_handoff", None)
    state.pop("_research_agent_idea_handoff_candidate_id", None)
    state["_active_main_page"] = "research_agent"
    state["_ra_view"] = "setup"
    state["_scroll_to_top"] = True
    if str(state.get("research_agent_question") or "").strip():
        state["_eu_ra_question_handoff_setup"] = True
    state["_assistant_notice"] = (
        "Research Agent is open in Idea exploration mode. It will call the backend idea-mining dry-run and stop at the proposed registry gate."
        if lang == "en" else
        "Research Agent 已打开 Idea 探索模式。它会调用底层 idea-mining dry-run，并停在 proposed registry 人工关口。"
    )
    state["_inline_ai_panel_open"] = False
    state["_floating_ai_open"] = False
    state["_sidebar_ai_open"] = False
    state.pop("_ai_pending_question", None)
    return seeded
