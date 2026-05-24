"""
EasyICU LLM Chat Assistant Module.

Provides an embedded conversational AI assistant that helps users
understand EasyICU features, interpret extraction results, and
answer ICU data analysis questions.

Supported providers:
        - HuggingFace (token required)
  - OpenAI, DeepSeek, Anthropic, OpenRouter, Together AI, Groq,
    SiliconFlow, and any custom OpenAI-compatible endpoint.

All API credentials are stored in session state only — never persisted.
"""

from __future__ import annotations

import ast
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import requests
import streamlit as st
from easyicu.webapp.components.constants import get_all_concepts
from easyicu.webapp.llm_config import (
    PROVIDERS,
    coerce_public_provider,
    ensure_llm_config_state,
    needs_api_key as _shared_needs_api_key,
    public_default_provider_key,
    public_provider_defaults,
    public_provider_keys,
)

_FEATURE_COUNT = len(get_all_concepts())

# ---------------------------------------------------------------------------
# System prompt — enriched with EasyICU documentation & concept catalogue
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You are an intelligent assistant embedded in **EasyICU**, an interactive platform \
for clinical data extraction and exploration across multiple public ICU databases.

## Platform Overview
EasyICU is a Python toolkit (v1.0) that provides:
- Unified access to **6 public ICU databases**: MIMIC-IV (miiv), MIMIC-III (mimic), \
eICU-CRD (eicu), AmsterdamUMCdb (aumc), HiRID (hirid), SICdb (sic)
- Automated extraction of **{_FEATURE_COUNT} standardized clinical concepts** across 19 feature modules
- A no-code Streamlit web interface for cohort construction, feature selection, \
quality review, and cohort comparison
- High-performance computing optimised for 16 GB RAM machines
- Export in Parquet, CSV, and XLSX formats; all parameters exportable as JSON for reproducibility

## Workflow (4 Steps)
1. **Data Source** — choose database & path (or Demo mode with simulated data)
2. **Cohort Selection** — filter by age, sex, ICU LOS, mortality, disease cohort (e.g. Sepsis-3, AKI, RRT, mechanical ventilation), and ICD keywords where supported
3. **Select Features** — pick from 19 modules ({_FEATURE_COUNT} concepts); supports SOFA-1, SOFA-2, \
Sepsis-3, KDIGO-AKI, circulatory failure, etc.
4. **Export Data** — batch export to disk; streaming architecture, subprocess memory isolation

## Main Web Areas
- **Tutorial** — workflow guide, usage examples, and the in-app data dictionary
- **Quick Visualization** — load extracted data and inspect trends, missingness, and distributions
- **Cohort Analysis** — compare groups and review downstream cohort summaries
- **AI Assistant** — guided help for navigation, feature planning, troubleshooting, and evidence lookup

## Feature Modules & Concepts (19 modules, {_FEATURE_COUNT} concepts)
- **Vital Signs** (8): hr, map, sbp, dbp, pulse_pressure, temp, spo2, resp
- **Respiratory** (14): pafi, safi, fio2, supp_o2, vent_ind, vent_start, vent_end, \
o2sat, sao2, mech_vent, ett_gcs, ecmo, ecmo_indication, adv_resp
- **Ventilator Parameters** (12): peep, tidal_vol, tidal_vol_set, pip, plateau_pres, \
mean_airway_pres, minute_vol, vent_rate, etco2, compliance, driving_pres, ps
- **Blood Gas** (9): be, cai, hbco, lact, methb, pco2, ph, po2, tco2
- **Chemistry** (22): alb, alp, alt, ast, anion_gap, bicar, bili, bili_dir, bun, ca, ck, ckmb, \
cl, crea, crp, glu, k, mg, na, phos, tnt, tri
- **Hematology** (20): bnd, basos, eos, esr, fgn, hba1c, hct, hgb, inr_pt, lymph, \
mch, mchc, mcv, neut, plt, pt, ptt, rbc, rdw, wbc
- **Vasopressors** (17): norepi_rate/dur/equiv/60, epi_rate/dur/60, \
dopa_rate/dur/60, dobu_rate/dur/60, adh_rate, phn_rate, vaso_ind, other_vaso
- **Medications** (15): abx, cort, dex, ins, amiodarone, dexmedetomidine, fentanyl, \
furosemide, heparin, mannitol, midazolam, milrinone, morphine, propofol, rocuronium
- **Renal / KDIGO** (17): urine, urine24, uo_6h/12h/24h, rrt, rrt_criteria, \
aki, aki_stage, aki_stage_creat/uo/rrt, creat_low_past_48hr/7day, uo_rt_6hr/12hr/24hr
- **Neurological** (11): avpu, egcs, gcs, mgcs, rass, tgcs, vgcs, sedated_gcs, \
motor_response, delirium_positive, delirium_tx
- **Circulatory** (3): mech_circ_support, circ_failure, circ_event
- **Demographics** (6): age, bmi, height, sex, weight, adm
- **Other Scores** (4): qsofa, sirs, mews, news
- **Outcome** (3): death, los_icu, los_hosp
- **SOFA-2 Score** (7): sofa2, sofa2_resp/coag/liver/cardio/cns/renal
- **SOFA-1 Score** (7): sofa, sofa_resp/coag/liver/cardio/cns/renal
- **Sepsis-3** (2): sep3_sofa1, sep3_sofa2
- **Sepsis Shared** (3): susp_inf, infection_icd, samp

## Python API Examples
```python
from easyicu import load_concepts, load_sofa, load_sofa2
# Load specific concepts
df = load_concepts(['hr','sbp','temp'], database='miiv', data_path='/data/miiv')
# Load SOFA scores
sofa = load_sofa(database='eicu', data_path='/data/eicu')
```

## Key Clinical Scoring
- **SOFA-2**: 2025 revised criteria implemented in EasyICU (6 sub-systems)
- **SOFA-1**: Original 1996 SOFA score
- **Sepsis-3**: Suspected infection + SOFA ≥ 2
- **KDIGO-AKI**: Stages 0-3 based on creatinine rise, urine output, and RRT
- **qSOFA**: Quick bedside screen (RR≥22, SBP≤100, altered mentation)

## Database Patient Counts (approx.)
| DB | Patients | ID column |
|----|----------|-----------|
| MIMIC-IV (miiv) | 94,458 | stay_id |
| MIMIC-III (mimic) | 61,532 | icustay_id |
| eICU-CRD (eicu) | 200,859 | patientunitstayid |
| AmsterdamUMCdb (aumc) | 23,106 | admissionid |
| HiRID (hirid) | 33,905 | patientid |
| SICdb (sic) | 27,386 | CaseID |

## Response Rules
- Respond in the same language the user is using.
- Be concise and practical. Start with the direct answer, then give short bullets only if useful.
- If the user describes a study goal or clinical task, answer in a task-first way:
  1. restate what EasyICU can support for that goal,
  2. suggest the recommended cohort definition,
  3. suggest which modules / concepts to extract,
  4. mention the key web steps to follow.
- Prioritize helping users use the EasyICU web interface and workflows. Code-level explanations are secondary unless the user explicitly asks for implementation details.
- When a user asks where something is in the web app, answer with the relevant page, step, or in-app action first. Do not default to repo files.
- When the user asks about EasyICU implementation, prefer the exact EasyICU concept names and outputs over generic medical summaries.
- Prefer suggesting concrete research workflows such as early warning, trajectory modelling, cohort construction, outcome analysis, sensitivity analysis, and cross-database feature planning.
- You will also receive a local EasyICU code snapshot at runtime. Use that code context when answering implementation or file-level questions.
- When pointing users to project files or docs, prefer clickable Markdown links.
- Do not invent version years, guideline dates, concept names, or unsupported claims. If uncertain, say what you can confirm from EasyICU and stop there.
- For Sepsis questions, anchor the answer to EasyICU outputs: `sep3_sofa1`, `sep3_sofa2`, `susp_inf`, `infection_icd`, `samp`, `sofa`, `sofa2`.
- Avoid padded closers like "有什么我可以帮你的吗？" unless the user asks for more detail."""


PROJECT_CONTEXT_FILES = [
    "src/easyicu/webapp/app.py",
    "src/easyicu/webapp/llm_chat.py",
    "src/easyicu/api.py",
    "src/easyicu/load_concepts.py",
    "src/easyicu/concept.py",
    "src/easyicu/concept_callbacks.py",
    "src/easyicu/sofa2.py",
    "src/easyicu/sepsis.py",
    "src/easyicu/scores.py",
    "src/easyicu/resources.py",
    "README.md",
]

PROJECT_LINK_FILES = [
    "README.md",
    "HOSTED_LLM.md",
    "src/easyicu/data/concept-dict.json",
    "src/easyicu/data/sofa2-dict.json",
    "src/easyicu/webapp/app.py",
    "src/easyicu/webapp/llm_chat.py",
    "src/easyicu/hosted_llm_server.py",
    "src/easyicu/api.py",
    "src/easyicu/load_concepts.py",
    "src/easyicu/concept.py",
    "src/easyicu/concept_callbacks.py",
    "src/easyicu/sofa2.py",
    "src/easyicu/sepsis.py",
]

ALL_PRESET_GROUP_KEYS = [
    "sofa2_score",
    "sofa1_score",
    "sepsis3_sofa2",
    "sepsis3_sofa1",
    "sepsis_shared",
    "vitals",
    "respiratory",
    "ventilator",
    "blood_gas",
    "chemistry",
    "hematology",
    "vasopressors",
    "medications",
    "renal",
    "neurological",
    "circulatory",
    "demographics",
    "other_scores",
    "outcome",
]

SEPSIS_PRESET_CONCEPTS = [
    "sep3_sofa2",
    "sep3_sofa1",
    "susp_inf",
    "infection_icd",
    "samp",
    "sofa2",
    "sofa2_resp",
    "sofa2_coag",
    "sofa2_liver",
    "sofa2_cardio",
    "sofa2_cns",
    "sofa2_renal",
    "sofa",
    "sofa_resp",
    "sofa_coag",
    "sofa_liver",
    "sofa_cardio",
    "sofa_cns",
    "sofa_renal",
    "abx",
    "cort",
    "lact",
    "hr",
    "map",
    "sbp",
    "temp",
    "resp",
    "spo2",
    "fio2",
    "pafi",
    "wbc",
    "crp",
    "crea",
    "bili",
    "urine",
    "urine24",
    "rrt",
    "death",
    "los_icu",
]


def _build_workflow_status_context(lang: str) -> str:
    """Summarize the current EasyICU web workflow state for the assistant."""
    entry_mode = st.session_state.get("entry_mode", "none")
    database = st.session_state.get("database", "miiv")
    data_path = (st.session_state.get("data_path") or "").strip()
    path_set = bool(data_path)
    path_validated = bool(st.session_state.get("path_validated", False))
    last_validation = st.session_state.get("last_validation", {}) or {}
    last_validated_path = st.session_state.get("last_validated_path", "")
    convert_needed = bool(
        path_set
        and data_path == last_validated_path
        and last_validation.get("can_convert")
        and not path_validated
    )
    step2_confirmed = bool(st.session_state.get("step2_confirmed", False))
    step3_confirmed = bool(st.session_state.get("step3_confirmed", False))
    selected_count = len(st.session_state.get("selected_concepts", []))
    loaded_count = len(st.session_state.get("loaded_concepts", {}))
    pending_preset = st.session_state.get("_assistant_pending_feature_preset")
    pending_db = pending_preset.get("database") if isinstance(pending_preset, dict) else None

    if lang == "en":
        lines = [
            "Current EasyICU workflow state:",
            f"- entry_mode: {entry_mode}",
            f"- database: {database}",
            f"- data_path_set: {'yes' if path_set else 'no'}",
            f"- data_path_validated: {'yes' if path_validated else 'no'}",
            f"- convert_or_setup_needed: {'yes' if convert_needed else 'no'}",
            f"- step2_confirmed: {'yes' if step2_confirmed else 'no'}",
            f"- step3_confirmed: {'yes' if step3_confirmed else 'no'}",
            f"- selected_concepts_count: {selected_count}",
            f"- loaded_concepts_count: {loaded_count}",
        ]
        if pending_db:
            lines.append(f"- pending_ai_preset_for_database: {pending_db}")
        lines.extend([
            "",
            "Guidance rules for this session:",
            "- For web workflow questions, first say what mode/page/step the user should be in.",
            "- If the user is not in Real Data mode and asks about extracting from public ICU databases, explicitly tell them to switch to Real Data mode first.",
            "- If data_path is not set, the next step is to fill the data path.",
            "- If data_path is set but not validated, the next step is Validate Data Path or Convert & Setup.",
            "- If conversion/setup is needed, warn that it may take time but normally only needs to be done once per dataset.",
            "- If step2 is not confirmed yet, tell the user to finish cohort selection before feature selection.",
            "- Only discuss code locations if the user explicitly asks about implementation.",
        ])
        return "\n".join(lines)

    lines = [
        "当前 EasyICU 工作流状态：",
        f"- entry_mode: {entry_mode}",
        f"- database: {database}",
        f"- 是否已填写数据路径: {'是' if path_set else '否'}",
        f"- 数据路径是否已验证: {'是' if path_validated else '否'}",
        f"- 是否需要转换/设置: {'是' if convert_needed else '否'}",
        f"- 步骤2是否已确认: {'是' if step2_confirmed else '否'}",
        f"- 步骤3是否已确认: {'是' if step3_confirmed else '否'}",
        f"- 当前已选特征数: {selected_count}",
        f"- 当前已加载概念数: {loaded_count}",
    ]
    if pending_db:
        lines.append(f"- 已挂起的 AI 预设数据库: {pending_db}")
    lines.extend([
        "",
        "本轮回答规则：",
        "- 对 Web 使用问题，先说明用户应该处于哪个模式、页面和步骤。",
        "- 如果用户问的是公共 ICU 数据库提取，而当前不是 Real Data 模式，要明确先切换到真实数据模式。",
        "- 如果还没填写数据路径，下一步就是先填写数据路径。",
        "- 如果数据路径已填写但还没验证，下一步就是点击验证数据路径或转换并设置。",
        "- 如果需要转换/设置，要提醒用户这一步可能较耗时，但通常同一份数据只需要做一次。",
        "- 如果步骤2还没确认，不要直接让用户去选特征；先让他完成队列筛选确认。",
        "- 只有当用户明确问实现细节时，才讨论代码位置。",
    ])
    return "\n".join(lines)


def _repo_blob_base() -> str:
    """Return the GitHub blob base used for clickable file links."""
    return os.getenv(
        "EASYICU_REPO_BLOB_BASE",
        "https://github.com/shen-lab-icu/easyicu/blob/main",
    ).rstrip("/")

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _init_chat_state():
    """Ensure all chat-related session keys exist."""
    ensure_llm_config_state()
    defaults = {
        "llm_enabled": False,
        "llm_provider": public_default_provider_key(),
        "llm_api_key": "",
        "llm_model": "",
        "llm_base_url": "",
        "llm_configured": False,
        "llm_messages": [],
        "llm_last_tool_events": [],
        "llm_last_verification": None,
        "_floating_ai_open": False,
        "_ai_pending_question": None,
        # Background response tracking
        "_ai_bg_responding": False,        # True while LLM is generating in background
        "_ai_bg_response_ready": False,    # True when background response finished
        "_ai_bg_unread_count": 0,          # Number of unread responses
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _sync_llm_toggle_before_render() -> None:
    """Synchronize the sidebar toggle before its widget is instantiated."""
    if "_llm_toggle" not in st.session_state or st.session_state.pop("_llm_toggle_sync_pending", False):
        st.session_state["_llm_toggle"] = bool(st.session_state.get("llm_enabled", False))


def _apply_floating_ai_toggle(enabled: bool) -> None:
    """Apply the legacy assistant toggle used by older sessions."""
    enabled = bool(enabled)
    st.session_state.llm_enabled = enabled
    st.session_state["_floating_ai_open"] = enabled
    if not enabled:
        st.session_state["_ai_pending_question"] = None


def _close_floating_ai_panel(*, disable_assistant: bool = False) -> None:
    """Close the floating panel, optionally turning off the sidebar toggle too."""
    st.session_state["_floating_ai_open"] = False
    st.session_state["_ai_pending_question"] = None
    if disable_assistant:
        st.session_state.llm_enabled = False
        st.session_state["_llm_toggle_sync_pending"] = True


def _repo_root() -> Path:
    """Return the project root based on this module location."""
    return Path(__file__).resolve().parents[3]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _extract_outline(path: Path, limit: int = 16) -> list[tuple[str, int]]:
    """Return top-level defs/classes with line numbers for quick code navigation."""
    try:
        tree = ast.parse(_read_text(path))
    except Exception:
        return []

    items = []
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            items.append((node.name, node.lineno))
    return items[:limit]


@lru_cache(maxsize=1)
def _load_project_index() -> dict[str, dict[str, object]]:
    """Build a lightweight local code index for code-aware chat answers."""
    root = _repo_root()
    index: dict[str, dict[str, object]] = {}
    for rel_path in PROJECT_CONTEXT_FILES:
        path = root / rel_path
        if not path.exists():
            continue
        text = _read_text(path)
        index[rel_path] = {
            "path": path,
            "text": text,
            "lines": text.splitlines(),
            "outline": _extract_outline(path),
        }
    return index


def _extract_identifiers(prompt: str) -> list[str]:
    """Extract likely code identifiers from user text."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_./-]{2,}", prompt or "")
    seen = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
    return seen[:8]


def _is_code_question(prompt: str) -> bool:
    """Heuristic to decide whether repo code context is useful."""
    prompt_l = (prompt or "").lower()
    keywords = (
        "code", "代码", "源码", "函数", "function", "class", "实现", "实现逻辑",
        "repo", "repository", "module", "文件", "file", "line", "行号",
        ".py", "app.py", "llm_chat", "load_concepts", "session_state",
        "bug", "fix", "stack trace", "traceback", "import", "api.py",
    )
    if any(word in prompt_l for word in keywords):
        return True

    identifiers = _extract_identifiers(prompt)
    return any(
        ("_" in tok) or ("." in tok) or ("/" in tok) or tok.lower().startswith("render_")
        for tok in identifiers
    )


def _make_snippet(lines: list[str], start_line: int, max_lines: int = 14) -> str:
    start_idx = max(start_line - 1, 0)
    end_idx = min(start_idx + max_lines, len(lines))
    snippet_lines = lines[start_idx:end_idx]
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(snippet_lines, start=start_idx))


def _build_project_context(prompt: str) -> str:
    """Build a compact local project snapshot for implementation questions."""
    index = _load_project_index()
    identifiers = [tok.lower() for tok in _extract_identifiers(prompt)]

    sections = [
        "Local EasyICU code snapshot:",
        "- You can answer based on this local repository snapshot.",
        "- If a detail is not covered below, say you only have partial local context.",
        "",
        "Key files:",
    ]
    for rel_path in PROJECT_CONTEXT_FILES[:8]:
        if rel_path in index:
            outline = index[rel_path]["outline"]
            preview = ", ".join(f"{name}@L{lineno}" for name, lineno in outline[:6])
            sections.append(f"- {rel_path}: {preview or 'file available'}")

    if not _is_code_question(prompt):
        return "\n".join(sections)

    matches = []
    for rel_path, meta in index.items():
        outline = meta["outline"]
        lines = meta["lines"]
        text_l = str(meta["text"]).lower()
        file_tokens = tuple(tok for tok in identifiers if "." in tok)

        for name, lineno in outline:
            name_l = name.lower()
            if any(tok in name_l or name_l in tok for tok in identifiers):
                matches.append(
                    (rel_path, name, lineno, _make_snippet(lines, lineno))
                )

        if file_tokens and rel_path.lower().endswith(file_tokens):
            matches.append((rel_path, rel_path, 1, _make_snippet(lines, 1, max_lines=18)))

        for tok in identifiers:
            if tok in text_l and len(matches) < 8:
                for idx, line in enumerate(lines, start=1):
                    if tok in line.lower():
                        matches.append((rel_path, f"match:{tok}", idx, _make_snippet(lines, idx)))
                        break

    if matches:
        sections.append("")
        sections.append("Relevant code excerpts:")
        seen = set()
        for rel_path, label, lineno, snippet in matches:
            key = (rel_path, lineno)
            if key in seen:
                continue
            seen.add(key)
            sections.append(f"[{rel_path}:{lineno}] {label}")
            sections.append("```python")
            sections.append(snippet)
            sections.append("```")
            if len(seen) >= 5:
                break

    return "\n".join(sections)


def _github_file_link(rel_path: str, label: str | None = None) -> str:
    """Build a clickable GitHub link for a project file."""
    clean_path = rel_path.strip().lstrip("/")
    text = label or Path(clean_path).name
    return f"[{text}]({_repo_blob_base()}/{clean_path})"


def _collect_quick_links(prompt: str, answer: str) -> list[tuple[str, str]]:
    """Collect the most relevant clickable links for the current answer."""
    prompt_l = (prompt or "").lower()
    answer_l = (answer or "").lower()
    combined = f"{prompt_l}\n{answer_l}"

    candidates: list[tuple[str, str]] = []
    seen_paths: set[str] = set()

    keyword_map = {
        "dictionary": "src/easyicu/data/concept-dict.json",
        "字典": "src/easyicu/data/concept-dict.json",
        "concept-dict": "src/easyicu/data/concept-dict.json",
        "sofa2-dict": "src/easyicu/data/sofa2-dict.json",
        "export": "src/easyicu/webapp/app.py",
        "app.py": "src/easyicu/webapp/app.py",
        "llm_chat": "src/easyicu/webapp/llm_chat.py",
        "agent": "src/easyicu/webapp/llm_chat.py",
        "hosted_llm_server": "src/easyicu/hosted_llm_server.py",
        "api.py": "src/easyicu/api.py",
        "load_concepts": "src/easyicu/load_concepts.py",
        "concept.py": "src/easyicu/concept.py",
        "sepsis": "src/easyicu/sepsis.py",
        "sofa": "src/easyicu/sofa2.py",
        "readme": "README.md",
    }

    for keyword, rel_path in keyword_map.items():
        if keyword in combined and rel_path not in seen_paths:
            seen_paths.add(rel_path)
            candidates.append((Path(rel_path).name, rel_path))

    for rel_path in PROJECT_LINK_FILES:
        if rel_path.lower() in combined and rel_path not in seen_paths:
            seen_paths.add(rel_path)
            candidates.append((Path(rel_path).name, rel_path))

    if not candidates:
        default_links = [
            ("README", "README.md"),
            ("Concept Dictionary", "src/easyicu/data/concept-dict.json"),
        ]
        for label, rel_path in default_links:
            if rel_path not in seen_paths:
                seen_paths.add(rel_path)
                candidates.append((label, rel_path))

    return candidates[:4]


def _append_quick_links(prompt: str, answer: str, lang: str) -> str:
    """Append code-file quick links, but only for explicit implementation questions."""
    if not answer.strip():
        return answer
    if not _is_code_question(prompt):
        return answer
    if "http://" in answer or "https://" in answer or "Quick links" in answer or "快捷链接" in answer:
        return answer

    links = _collect_quick_links(prompt, answer)
    if not links:
        return answer

    title = "Quick links" if lang == "en" else "快捷链接"
    lines = [answer.rstrip(), "", f"**{title}**"]
    for label, rel_path in links:
        lines.append(f"- {_github_file_link(rel_path, label)}")
    return "\n".join(lines)


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


def _suggest_ui_actions(prompt: str, answer: str, lang: str) -> list[dict[str, object]]:
    """Suggest in-app navigation or preset actions."""
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

    dictionary_requested = any(key in prompt_l for key in ["字典", "数据字典", "dictionary", "feature list", "concept list"])
    tutorial_requested = any(key in prompt_l for key in ["tutorial", "教程", "step", "步骤", "how do i", "怎么做", "workflow", "流程", "guide", "使用"])
    viz_requested = any(key in prompt_l for key in [
        "quick visualization", "快速可视化", "load data", "加载数据", "visualization",
        "visualize", "visualise", "plot", "可视化", "图表", "数据分析", "分析我的数据",
    ])
    cohort_requested = any(key in prompt_l for key in ["cohort", "队列", "compare", "comparison", "dashboard", "仪表板"])
    export_requested = any(key in prompt_l for key in ["export", "导出"])

    if dictionary_requested or (
        any(key in answer_l for key in ["data dictionary", "数据字典", "concept dictionary"]) and
        any(key in prompt_l for key in ["where", "在哪", "在哪里", "怎么找", "how to find", "查看"])
    ):
        add_nav("home_dict", "📖 Open Data Dictionary", "📖 打开数据字典")

    if tutorial_requested and not dictionary_requested:
        add_nav("tutorial", "📚 Open Tutorial", "📚 打开教程")

    if viz_requested:
        add_nav("viz", "📊 Open Quick Visualization", "📊 前往快速可视化")

    if cohort_requested:
        add_nav("cohort", "🔬 Open Cohort Analysis", "🔬 前往队列分析")

    if export_requested:
        add_nav("tutorial", "📚 Open Export Guide", "📚 打开导出教程")

    target_db = _infer_db_from_text(combined)
    is_all_features_request = any(key in combined for key in [
        "所有临床指标", "所有特征", "全部特征", "all clinical features",
        "all concepts", "all indicators", "all features",
    ])
    is_sepsis_extract_request = (
        any(key in combined for key in ["sepsis", "脓毒症", "septic shock", "脓毒性休克"])
        and any(key in combined for key in ["提取", "抽取", "export", "导出", "select", "选择"])
    )

    if target_db == "miiv" and is_all_features_request:
        add_preset(
            "preset_miiv_all",
            "⚙️ Prepare MIMIC-IV Full Feature Selection",
            "⚙️ 预设 MIMIC-IV 全量特征选择",
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
            "🦠 Prepare MIMIC-IV Sepsis Feature Set",
            "🦠 预设 MIMIC-IV Sepsis 特征集",
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


def _render_nav_actions(actions: list[dict[str, object]], key_prefix: str) -> None:
    """Render in-app navigation and preset actions as buttons."""
    if not actions:
        return
    action_cols = st.columns(len(actions))
    for action_idx, action in enumerate(actions):
        with action_cols[action_idx]:
            if st.button(
                action["label"],
                key=f"{key_prefix}_{action_idx}_{action['id']}",
                use_container_width=True,
            ):
                if action.get("kind") == "preset":
                    st.session_state["_assistant_preset_request"] = dict(action.get("payload") or {})
                    st.session_state["_scroll_to_tab"] = str(action.get("scroll_to") or "tutorial")
                else:
                    st.session_state["_scroll_to_tab"] = str(action["id"])
                st.rerun()


def _is_external_lookup_question(prompt: str) -> bool:
    """Heuristic to decide whether authoritative web lookup is appropriate."""
    prompt_l = (prompt or "").lower()
    evidence_terms = (
        "pubmed", "pmid", "文献", "论文", "reference", "references", "citation",
        "cite", "source", "sources", "链接", "证据", "指南", "guideline",
    )
    medical_terms = (
        "sepsis", "septic", "sofa", "kdigo", "aki", "icu", "vasopressor",
        "ventilation", "ecmo", "delirium", "mortality", "infection", "qsofa",
        "sirs", "脓毒症", "感染", "休克", "肾损伤", "指南", "呼吸机", "机械通气",
    )
    if any(term in prompt_l for term in evidence_terms):
        return True
    return any(term in prompt_l for term in medical_terms) and not _is_code_question(prompt)


def _search_pubmed(prompt: str, max_results: int = 3) -> tuple[list[dict[str, str]], str | None]:
    """Search PubMed via E-utilities and return top article metadata."""
    query = (prompt or "").strip()
    if not query:
        return [], None

    try:
        search_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "retmode": "json",
                "retmax": max_results,
                "sort": "relevance",
                "term": query,
            },
            timeout=15,
        )
        search_resp.raise_for_status()
        id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return [], None

        summary_resp = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={
                "db": "pubmed",
                "retmode": "json",
                "id": ",".join(id_list),
            },
            timeout=15,
        )
        summary_resp.raise_for_status()
        result = summary_resp.json().get("result", {})

        articles = []
        for pmid in id_list:
            item = result.get(pmid, {})
            authors = item.get("authors", [])[:3]
            author_text = ", ".join(author.get("name", "") for author in authors if author.get("name"))
            articles.append({
                "pmid": pmid,
                "title": item.get("title", "").strip().rstrip("."),
                "journal": item.get("fulljournalname") or item.get("source", ""),
                "pubdate": item.get("pubdate", ""),
                "authors": author_text,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
        return articles, query
    except requests.RequestException as exc:
        return [], f"PubMed lookup failed: {exc}"


def _build_pubmed_context(prompt: str) -> tuple[str, list[dict[str, str]]]:
    """Build authoritative medical context from PubMed results."""
    articles, info = _search_pubmed(prompt)
    events: list[dict[str, str]] = []
    if not articles:
        if info:
            events.append({"tool": "pubmed_search", "status": "error", "detail": info})
        return "", events

    events.append({
        "tool": "pubmed_search",
        "status": "ok",
        "detail": f"Query: {info}; results: {len(articles)}",
    })

    lines = [
        "Authoritative external sources retrieved from PubMed:",
        "- Use these only for medical/scientific claims, not for EasyICU implementation details.",
    ]
    for art in articles:
        meta = " | ".join(part for part in [art["authors"], art["journal"], art["pubdate"]] if part)
        lines.append(f"- PMID {art['pmid']}: {art['title']}")
        if meta:
            lines.append(f"  Meta: {meta}")
        lines.append(f"  URL: {art['url']}")
    return "\n".join(lines), events


def _compose_agent_messages(prompt: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Assemble system/context messages and return tool activity log."""
    tool_events: list[dict[str, str]] = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                "Answer format requirements:\n"
                "- For web-usage questions, lead with the exact page, step, or action in EasyICU.\n"
                "- After the direct answer, give the next 1-3 concrete UI steps the user should take now.\n"
                "- Mention files/functions only when the user explicitly asks about implementation.\n"
                "- If local code context was used, mention the relevant file/function names.\n"
                "- If external medical sources were used, include a `Sources` section with direct URLs.\n"
                "- Do not reveal hidden chain-of-thought or private deliberation."
            ),
        },
        {"role": "system", "content": _build_workflow_status_context(st.session_state.get("language", "en"))},
    ]

    project_context = _build_project_context(prompt) if _is_code_question(prompt) else ""
    if project_context:
        tool_events.append({
            "tool": "local_code_search",
            "status": "ok",
            "detail": "Built local EasyICU code snapshot",
        })
        messages.append({"role": "system", "content": project_context})

    if _is_external_lookup_question(prompt):
        pubmed_context, pubmed_events = _build_pubmed_context(prompt)
        tool_events.extend(pubmed_events)
        if pubmed_context:
            messages.append({"role": "system", "content": pubmed_context})

    messages.extend(st.session_state.llm_messages)
    return messages, tool_events


def _stream_text(stream, placeholder):
    """Render streaming text manually."""
    chunks = []
    for token in _token_generator(stream):
        chunks.append(token)
        visible_text = _strip_llm_reasoning("".join(chunks))
        placeholder.markdown((visible_text + "▌") if visible_text else "…")
    text = _strip_llm_reasoning("".join(chunks))
    placeholder.markdown(text)
    return text


def _strip_llm_reasoning(text: str) -> str:
    """Remove model-private reasoning blocks from OpenAI-compatible outputs."""
    if not text:
        return ""
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.I | re.S)
    cleaned = re.sub(r"<think\b[^>]*>.*$", "", cleaned, flags=re.I | re.S)
    return cleaned.strip()


def _parse_verification_report(text: str) -> dict[str, object]:
    """Parse verifier output into a structured result."""
    text = _strip_llm_reasoning(text)
    result = {
        "status": "uncertain",
        "issues": [],
        "corrected_answer": "",
        "raw": text.strip(),
    }
    if not text:
        return result

    status_match = re.search(r"STATUS:\s*(pass|corrected|uncertain)", text, re.I)
    if status_match:
        result["status"] = status_match.group(1).lower()

    issues_match = re.search(r"ISSUES:\s*(.*?)\nCORRECTED_ANSWER:", text, re.S | re.I)
    if issues_match:
        issue_block = issues_match.group(1).strip()
        result["issues"] = [
            line.lstrip("-* ").strip()
            for line in issue_block.splitlines()
            if line.strip()
        ]

    corrected_match = re.search(r"CORRECTED_ANSWER:\s*(.*)$", text, re.S | re.I)
    if corrected_match:
        result["corrected_answer"] = corrected_match.group(1).strip()

    return result


def _verify_response(client, messages: list[dict[str, str]], draft: str, lang: str) -> dict[str, object]:
    """Run a second-pass verifier against the generated draft."""
    verifier_prompt = (
        "You are a strict medical and technical answer verifier.\n"
        "Check the assistant draft against the provided EasyICU code context and any PubMed evidence.\n"
        "Rules:\n"
        "- Mark `pass` only if the draft is supported and appropriately cautious.\n"
        "- Mark `corrected` if you rewrote any unsupported, vague, or overclaimed content.\n"
        "- Mark `uncertain` if the evidence is insufficient.\n"
        "- Preserve links and concrete file/function references when valid.\n"
        "- For web workflow questions, make sure the answer tells the user the next concrete UI step.\n"
        "- Do not reveal hidden chain-of-thought.\n"
        "- Return exactly this format:\n"
        "STATUS: <pass|corrected|uncertain>\n"
        "ISSUES:\n"
        "- <issue 1>\n"
        "CORRECTED_ANSWER:\n"
        "<final answer>\n"
    )
    verify_messages = [
        {"role": "system", "content": verifier_prompt},
        *messages,
        {"role": "assistant", "content": draft},
    ]
    try:
        response = client.chat.completions.create(
            model=st.session_state.get("llm_model", "").strip()
            or public_provider_defaults(st.session_state.get("llm_provider", public_default_provider_key()))[2],
            messages=verify_messages,
            stream=False,
        )
        text = response.choices[0].message.content if response.choices else ""
        parsed = _parse_verification_report(text)
        if not parsed.get("corrected_answer"):
            parsed["corrected_answer"] = draft
        return parsed
    except Exception as exc:
        return {
            "status": "uncertain",
            "issues": [
                ("Verification failed: " if lang == "en" else "校验失败：") + str(exc)
            ],
            "corrected_answer": draft,
            "raw": "",
        }


def _needs_api_key(provider: str) -> bool:
    return _shared_needs_api_key(provider)


def _is_configured() -> bool:
    """Return True when the provider is ready to use."""
    provider = st.session_state.get("llm_provider", public_default_provider_key())
    _, default_url, _default_model, _needs_key, _desc_en, _desc_zh = public_provider_defaults(provider)
    if _needs_api_key(provider) and not st.session_state.get("llm_api_key", "").strip():
        return False
    return bool((st.session_state.get("llm_base_url", "") or default_url).strip())


def _get_client():
    """Build and return an OpenAI-compatible client, or *None* on error."""
    try:
        from openai import OpenAI
        import httpx
    except ImportError:
        return None

    provider = coerce_public_provider(st.session_state.get("llm_provider", public_default_provider_key()))
    _display, default_url, _default_model, _needs_key, _desc_en, _desc_zh = public_provider_defaults(provider)
    api_key = st.session_state.get("llm_api_key", "").strip()
    base_url = st.session_state.get("llm_base_url", "").strip() or default_url or None

    if not api_key and not _needs_api_key(provider):
        api_key = os.getenv("EASYICU_HOSTED_CLIENT_TOKEN", "easyicu-hosted")

    if not api_key:
        return None

    # Ignore system proxy env vars by default to avoid optional SOCKS dependency
    # failures inside embedded Streamlit sessions.
    http_client = httpx.Client(
        timeout=120.0,
        trust_env=False,
        follow_redirects=True,
    )
    return OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)


# ---------------------------------------------------------------------------
# UI — Settings panel (rendered inside an expander in the sidebar)
# ---------------------------------------------------------------------------

def render_llm_settings():
    """Render LLM configuration controls in the sidebar."""
    _init_chat_state()
    _sync_llm_toggle_before_render()
    lang = st.session_state.get("language", "en")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] .easyicu-ai-sidebar-card {
            display: grid;
            grid-template-columns: 2.2rem 1fr;
            gap: 0.62rem;
            align-items: center;
            padding: 0.72rem 0.82rem;
            border-radius: 14px;
            border: 1px solid #cfe0f3;
            background: linear-gradient(135deg, #ffffff 0%, #f5f9ff 100%);
            box-shadow: 0 9px 22px rgba(15, 23, 42, 0.05);
            margin: 0.24rem 0 0.18rem;
        }
        [data-testid="stSidebar"] .easyicu-ai-sidebar-avatar {
            width: 2.15rem;
            height: 2.15rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            background: linear-gradient(135deg, #f59e0b 0%, #f97316 68%, #2563eb 100%);
            box-shadow: 0 9px 18px rgba(249, 115, 22, 0.2);
            font-size: 1rem;
        }
        [data-testid="stSidebar"] .easyicu-ai-sidebar-title {
            color: #0b1f44;
            font-size: 0.9rem;
            font-weight: 900;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }
        [data-testid="stSidebar"] .easyicu-ai-sidebar-subtitle {
            color: #64748b;
            font-size: 0.72rem;
            line-height: 1.38;
            margin-top: 0.12rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    enabled = bool(st.session_state.llm_enabled)
    status_title = "AI Assistant" if lang == "en" else "AI 助手"
    status_subtitle = (
        "Embedded guidance for the current EasyICU workflow."
        if lang == "en" else
        "嵌入当前 EasyICU 工作流的页面感知式助手。"
    )
    if enabled and _is_configured():
        status_subtitle = (
            "Embedded chat is ready in this panel."
            if lang == "en" else
            "嵌入式聊天已在此面板中就绪。"
        )
    st.markdown(
        f"""
        <div class="easyicu-ai-sidebar-card">
            <div class="easyicu-ai-sidebar-avatar">🤖</div>
            <div>
                <div class="easyicu-ai-sidebar-title">{status_title}</div>
                <div class="easyicu-ai-sidebar-subtitle">{status_subtitle}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    label = "⚙️ AI settings" if lang == "en" else "⚙️ AI 设置"
    with st.expander(label, expanded=False):
        previous_enabled = bool(st.session_state.llm_enabled)
        enabled = st.toggle(
            "Enable AI assistant" if lang == "en" else "启用 AI 助手",
            value=previous_enabled,
            key="_llm_toggle",
        )
        st.session_state.llm_enabled = bool(enabled)
        st.session_state["_floating_ai_open"] = False
        if enabled:
            st.session_state["_sidebar_ai_open"] = True
        if not enabled:
            hint = ("Enable this to use the AI panel in the main workspace."
                    if lang == "en"
                    else "启用后可在主工作区使用 AI 面板。")
            st.caption(hint)
            return

        # Provider
        provider_keys = public_provider_keys()
        current_provider = coerce_public_provider(st.session_state.get("llm_provider", public_default_provider_key()))
        idx = provider_keys.index(current_provider) if current_provider in provider_keys else 0
        provider = st.selectbox(
            "Provider" if lang == "en" else "服务商",
            options=provider_keys,
            index=idx,
            format_func=lambda k: public_provider_defaults(k)[0],
            key="_llm_provider_sel",
        )
        st.session_state.llm_provider = provider

        # Provider description
        p_info = public_provider_defaults(provider)
        desc = p_info[4] if lang == "en" else p_info[5]
        st.caption(desc)

        needs_key = _needs_api_key(provider)

        if needs_key:
            # API Key (password field)
            api_key = st.text_input(
                "API Key",
                value=st.session_state.llm_api_key,
                type="password",
                key="_llm_api_key_inp",
                placeholder="sk-...",
            )
            st.session_state.llm_api_key = api_key

        # Base URL
        _, default_url, default_model, _, _, _ = p_info
        base_url = st.text_input(
            "API Base URL",
            value=st.session_state.llm_base_url or default_url,
            key="_llm_base_url_inp",
            help="Leave default for standard providers" if lang == "en"
                 else "标准服务商保持默认即可",
        )
        st.session_state.llm_base_url = base_url

        # Model
        model = st.text_input(
            "Model" if lang == "en" else "模型名称",
            value=st.session_state.llm_model or default_model,
            key="_llm_model_inp",
            placeholder=default_model or "model-name",
        )
        st.session_state.llm_model = model

        # Status indicator
        if _is_configured():
            st.success("✅ " + ("Ready" if lang == 'en' else "已就绪"))
            st.session_state.llm_configured = True
        else:
            st.warning("⚠️ " + ("Enter API Key to enable chat"
                                 if lang == 'en' else "请输入 API Key"))
            st.session_state.llm_configured = False

        st.caption(
            "Open the AI panel in the main workspace for page-aware guidance."
            if lang == "en" else
            "请在主工作区打开 AI 面板，以获得页面感知式建议。"
        )


def _build_chat_export_text() -> str:
    """Serialize chat history to markdown."""
    lines = ["# EasyICU AI Chat Export", ""]
    for msg in st.session_state.get("llm_messages", []):
        role = "User" if msg.get("role") == "user" else "Assistant"
        lines.append(f"## {role}")
        lines.append(msg.get("content", "").strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# UI — Chat tab (rendered as a main-area tab)
# ---------------------------------------------------------------------------

def render_chat_tab():
    """Render the full chat interface inside a main tab."""
    _init_chat_state()
    lang = st.session_state.get("language", "en")

    # ---- Guard: not enabled --------------------------------------------------
    if not st.session_state.llm_enabled:
        st.info(
            "🤖 " + (
                "The AI Assistant is disabled. Enable it in the sidebar **🤖 AI Assistant** section."
                if lang == "en" else
                "AI 助手当前已关闭。请在侧边栏 **🤖 AI 助手** 中开启。"
            )
        )
        # Show a brief intro even when disabled
        _render_intro(lang)
        return

    # ---- Guard: not configured -----------------------------------------------
    if not _is_configured():
        st.warning(
            "🔑 " + (
                "Please configure your API Key in the sidebar **🤖 AI Assistant** section first."
                if lang == "en" else
                "请先在侧边栏 **🤖 AI 助手** 中设置 API Key。"
            )
        )
        return

    # ---- Guard: openai package missing ---------------------------------------
    try:
        import openai as _openai_mod  # noqa: F401
    except ImportError:
        st.error(
            "📦 " + (
                "The `openai` Python package is required. "
                "Install it with: `pip install openai`"
                if lang == "en" else
                "需要安装 `openai` Python 包。请执行: `pip install openai`"
            )
        )
        return

    # ---- Header --------------------------------------------------------------
    provider_name = public_provider_defaults(st.session_state.get("llm_provider", public_default_provider_key()))[0]
    model_name = (st.session_state.get("llm_model", "").strip()
                  or public_provider_defaults(st.session_state.get("llm_provider", public_default_provider_key()))[2]
                  or "—")
    st.markdown(
        "##### " + ("💬 Chat with AI Assistant" if lang == "en"
                     else "💬 与 AI 助手对话")
    )
    st.caption(f"**{provider_name}** · `{model_name}`")
    st.caption(
        "Guided mode: web workflow first, then code and evidence when needed"
        if lang == "en" else
        "引导模式：优先教你如何使用 Web 工作流，再按需补充代码与证据"
    )

    with st.expander("💡 " + ("What can I ask?" if lang == "en" else "我可以问什么？"),
                      expanded=False):
        _render_tips(lang)

    if st.session_state.get("llm_last_tool_events"):
        with st.expander("🛠️ " + ("Last tool activity" if lang == "en" else "上次工具调用"), expanded=False):
            for event in st.session_state.llm_last_tool_events:
                icon = "✅" if event.get("status") == "ok" else "⚠️"
                st.markdown(f"{icon} `{event.get('tool', 'tool')}` — {event.get('detail', '')}")

    # ---- Render message history -----------------------------------------------
    history_container = st.container(height=680, border=True)
    with history_container:
        for msg_idx, msg in enumerate(st.session_state.llm_messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("actions"):
                    _render_nav_actions(msg["actions"], key_prefix=f"_llm_action_{msg_idx}")

        pending_prompt = st.session_state.pop("_ai_pending_question", None)
        if pending_prompt:
            _submit_prompt(pending_prompt, lang, history_container, key_prefix="_llm_tab")

    # ---- Chat input -----------------------------------------------------------
    placeholder = ("Ask a question about EasyICU …"
                    if lang == "en" else "输入关于 EasyICU 的问题 …")
    if prompt := st.chat_input(placeholder, key="_llm_chat_input"):
        _submit_prompt(prompt, lang, history_container, key_prefix="_llm_tab")


def _submit_prompt(prompt: str, lang: str, history_container, key_prefix: str = "_llm"):
    """Append a prompt, render local instant replies, or stream the model response."""
    prompt = (prompt or "").strip()
    if not prompt:
        return

    st.session_state.llm_messages.append({"role": "user", "content": prompt})
    with history_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    instant_reply = _get_instant_reply(prompt, lang)
    if instant_reply is not None:
        st.session_state.llm_last_tool_events = []
        st.session_state.llm_last_verification = {
            "status": "pass",
            "issues": [],
        }
        instant_actions = _suggest_ui_actions(prompt, instant_reply, lang)
        st.session_state.llm_messages.append(
            {
                "role": "assistant",
                "content": instant_reply,
                "actions": instant_actions,
            }
        )
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(instant_reply)
                _render_nav_actions(instant_actions, key_prefix=f"{key_prefix}_instant")
        return

    prep_placeholder = st.empty()
    prep_placeholder.info(
        "🛠️ Preparing tools..." if lang == "en" else "🛠️ 正在准备工具..."
    )
    try:
        messages, tool_events = _compose_agent_messages(prompt)
    except Exception as exc:
        prep_placeholder.empty()
        error_message = _handle_api_error(exc, lang, render=False)
        st.session_state.llm_messages.append(
            {"role": "assistant", "content": error_message, "actions": []}
        )
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(error_message)
        return
    st.session_state.llm_last_tool_events = tool_events
    prep_placeholder.empty()

    with history_container:
        with st.chat_message("assistant"):
            _stream_response(messages, lang)


def render_sidebar_chat_widget():
    """Render a compact embedded AI chat box in the sidebar."""
    _init_chat_state()
    lang = st.session_state.get("language", "en")
    pending_prompt = st.session_state.get("_ai_pending_question")
    expanded = bool(st.session_state.get("_sidebar_ai_open") or pending_prompt)

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-welcome {
            border: 1px solid #d8e2ee;
            border-left: 3px solid #0e7490;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.78rem 0.82rem;
            margin: 0.25rem 0 0.55rem;
            box-shadow: none;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-welcome-title {
            color: #111827;
            font-size: 0.84rem;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 0.28rem;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-welcome-subtitle,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-user-bubble,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-answer-card {
            color: #475569;
            font-size: 0.74rem;
            line-height: 1.45;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-sample {
            display: grid;
            gap: 0.38rem;
            margin: 0.52rem 0;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-user-bubble,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-answer-card,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-recommendation {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            background: #f8fafc;
            padding: 0.52rem 0.58rem;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-recommendation span {
            display: block;
            color: #64748b;
            font-size: 0.62rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.16rem;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-recommendation strong,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .floating-ai-welcome-hint {
            color: #0f172a;
            font-size: 0.72rem;
            line-height: 1.38;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel [data-testid="stChatMessageContent"] p,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel input,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel label {
            font-size: 0.75rem !important;
            line-height: 1.42;
        }
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel .stButton > button,
        [data-testid="stSidebar"] div.st-key-embedded_ai_chat_panel form button {
            min-height: 32px;
            border-radius: 7px;
            font-size: 0.74rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    title = "💬 Embedded chat" if lang == "en" else "💬 嵌入式对话"
    with st.expander(title, expanded=expanded):
        st.session_state["_sidebar_ai_open"] = expanded
        if not st.session_state.llm_enabled:
            st.caption(
                "Enable AI Assistant above to start chatting here."
                if lang == "en" else
                "请先在上方开启 AI 助手，然后在这里对话。"
            )
            return

        if not _is_configured():
            st.caption(
                "Configure the provider/API key above, then chat here."
                if lang == "en" else
                "请先在上方完成服务商/API Key 配置，再在这里对话。"
            )
            return

        with st.container(key="embedded_ai_chat_panel"):
            _render_compact_chat_panel(
                lang=lang,
                panel_key="_llm_sidebar_embedded",
                history_height=300,
                show_starters=not bool(pending_prompt),
            )


def render_inline_ai_panel():
    """Render the non-floating AI assistant as part of the main workspace."""
    _init_chat_state()
    lang = st.session_state.get("language", "en")
    pending_prompt = bool(st.session_state.get("_ai_pending_question"))
    if pending_prompt:
        st.session_state["_inline_ai_panel_open"] = True
        st.session_state["llm_enabled"] = True
        st.session_state["_llm_toggle"] = True
    if not st.session_state.get("_inline_ai_panel_open", False):
        return

    st.session_state["_floating_ai_open"] = False
    st.markdown(
        """
        <style>
        .stApp div.st-key-inline_ai_assistant_panel {
            border: 1px solid #d6dee8;
            border-left: 3px solid #0e7490;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.72rem 0.78rem 0.78rem;
            margin: 0.72rem 0 1rem;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-header {
            display: flex;
            align-items: center;
            gap: 0.62rem;
            min-width: 0;
            padding: 0.04rem 0 0.35rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-avatar {
            width: 2rem;
            height: 2rem;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #0f172a;
            background: #ecfeff;
            border: 1px solid #a5f3fc;
            font-size: 1rem;
            flex: 0 0 auto;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-title {
            color: #0f172a;
            font-size: 0.95rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-subtitle {
            color: #64748b;
            font-size: 0.76rem;
            line-height: 1.35;
            margin-top: 0.1rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-welcome {
            border: 1px solid #dbe4ee;
            border-radius: 8px;
            background: #f8fafc;
            padding: 0.8rem 0.88rem;
            margin-bottom: 0.62rem;
            box-shadow: none;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-welcome-title {
            color: #0f172a;
            font-size: 0.92rem;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 0.28rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-welcome-subtitle,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-user-bubble,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-answer-card {
            color: #475569;
            font-size: 0.82rem;
            line-height: 1.5;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-sample {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 0.5rem;
            margin: 0.58rem 0 0.62rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-user-bubble,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-answer-card,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            background: #ffffff;
            padding: 0.58rem 0.64rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation {
            grid-column: 1 / -1;
            display: grid;
            gap: 0.12rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation span {
            color: #64748b;
            font-size: 0.68rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation strong,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-welcome-hint {
            color: #0f172a;
            font-size: 0.8rem;
            line-height: 1.38;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] p,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stMarkdownContainer"] p,
        .stApp div.st-key-inline_ai_assistant_panel input,
        .stApp div.st-key-inline_ai_assistant_panel label {
            font-size: 0.84rem !important;
            line-height: 1.48;
        }
        .stApp div.st-key-inline_ai_assistant_panel .stButton > button,
        .stApp div.st-key-inline_ai_assistant_panel form button {
            min-height: 34px;
            border-radius: 7px;
            font-size: 0.8rem;
        }
        @media (max-width: 900px) {
            .stApp div.st-key-inline_ai_assistant_panel .floating-ai-sample {
                grid-template-columns: 1fr;
            }
            .stApp div.st-key-inline_ai_assistant_panel {
                margin-top: 0.55rem;
                padding: 0.62rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="inline_ai_assistant_panel"):
        left, close_col = st.columns([10, 1], gap="small")
        with left:
            title = "AI assistant" if lang == "en" else "AI 助手"
            subtitle = (
                "Page-aware guidance embedded in the current workspace."
                if lang == "en" else
                "嵌入当前工作区的页面感知式建议。"
            )
            st.markdown(
                f"""
                <div class="inline-ai-header">
                    <div class="inline-ai-avatar">AI</div>
                    <div>
                        <div class="inline-ai-title">{title}</div>
                        <div class="inline-ai-subtitle">{subtitle}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with close_col:
            if st.button(
                "",
                icon=":material/close:",
                key="_inline_ai_close",
                help="Close AI panel" if lang == "en" else "关闭 AI 面板",
                use_container_width=True,
            ):
                st.session_state["_inline_ai_panel_open"] = False
                st.session_state["_ai_pending_question"] = None
                st.rerun()

        if not st.session_state.get("llm_enabled", False):
            st.caption(
                "Enable AI Assistant in sidebar settings, then return here."
                if lang == "en" else
                "请先在侧栏设置中启用 AI 助手，然后回到这里。"
            )
            return

        if not _is_configured():
            st.caption(
                "Configure provider/API key in sidebar settings first."
                if lang == "en" else
                "请先在侧栏设置中配置服务商/API Key。"
            )
            return

        _render_compact_chat_panel(
            lang=lang,
            panel_key="_llm_inline_workspace",
            history_height=390,
            show_starters=not pending_prompt,
        )


def _starter_prompts(lang: str) -> list[str]:
    if lang == "en":
        return [
            "I want to build a sepsis early-warning cohort. Which EasyICU steps and modules should I use?",
            "I want to cluster sepsis patient trajectories over time. Which concepts and scores should I export?",
            "I want an AKI cohort for outcome analysis. How should I configure Step 2 and Step 3?",
            "I want an ICD-defined pneumonia or heart-failure cohort in MIMIC-IV. How should I use the disease template and ICD filter?",
        ]
    return [
        "我想构建脓毒症实时预警队列。EasyICU 里应该走哪些步骤、提取哪些模块？",
        "我想做脓毒症患者时间轨迹聚类。应该导出哪些时间序列概念和评分？",
        "我想做 AKI 队列结局分析。Step 2 和 Step 3 应该怎么配置？",
        "我想在 MIMIC-IV 里按 ICD 构建肺炎或心衰队列。疾病模板和 ICD filter 应该怎么用？",
    ]


def _render_chat_welcome(*, lang: str, panel_key: str, history_container, show_starters: bool = True) -> None:
    title = "Page-aware assistant guidance" if lang == "en" else "页面感知式 AI 引导"
    subtitle = (
        "Grounded in your current page, selected database, cohort filters, and feature modules."
        if lang == "en" else
        "基于当前页面、数据库、队列筛选和特征模块给出下一步建议。"
    )
    prompt_hint = (
        "Try one of these questions:" if lang == "en" else "你可以直接点下面这些问题："
    ) if show_starters else (
        "Ask about your current workflow, cohort, or export settings."
        if lang == "en" else
        "可以直接询问当前流程、队列筛选或导出设置。"
    )
    sample_q = (
        "Which setting should I check before exporting SOFA features?"
        if lang == "en" else
        "导出 SOFA 特征前，我应该重点检查哪个设置？"
    )
    sample_a = (
        "Check that the cohort is confirmed, SOFA-1/SOFA-2 concepts are selected, and the export path is writable."
        if lang == "en" else
        "请确认队列已经锁定、SOFA-1/SOFA-2 概念已选择，并且导出路径可写。"
    )
    rec_label = "Recommended next step" if lang == "en" else "推荐下一步"
    rec_text = (
        "Open feature selection and verify score modules before export."
        if lang == "en" else
        "进入特征选择页，在导出前检查评分模块。"
    )

    st.markdown(
        f'''
        <div class="floating-ai-welcome">
            <div class="floating-ai-welcome-title">{title}</div>
            <div class="floating-ai-welcome-subtitle">{subtitle}</div>
            <div class="floating-ai-sample">
                <div class="floating-ai-user-bubble">{sample_q}</div>
                <div class="floating-ai-answer-card">{sample_a}</div>
                <div class="floating-ai-recommendation">
                    <span>{rec_label}</span>
                    <strong>{rec_text}</strong>
                </div>
            </div>
            <div class="floating-ai-welcome-hint">{prompt_hint}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    if show_starters:
        prompts = _starter_prompts(lang)
        for idx, starter in enumerate(prompts):
            if st.button(
                starter,
                key=f"{panel_key}_starter_{idx}",
                use_container_width=True,
            ):
                _submit_prompt(starter, lang, history_container, key_prefix=f"{panel_key}_starter")
                st.rerun()


def _render_compact_chat_panel(*, lang: str, panel_key: str, history_height: int = 320, show_starters: bool = True) -> None:
    """Render a compact chat history + input form panel."""
    # Check for background response that completed while panel was minimized
    bg_result = _check_bg_response()
    if bg_result:
        st.session_state["_ai_bg_responding"] = False
        st.session_state["_ai_bg_session_id"] = None
        if bg_result.get("status") == "done":
            final_answer = bg_result["answer"]
            # Apply post-processing
            if st.session_state.llm_messages:
                last_user_msg = ""
                for m in reversed(st.session_state.llm_messages):
                    if m["role"] == "user":
                        last_user_msg = m["content"]
                        break
                final_answer = _append_quick_links(last_user_msg, final_answer, lang)
                response_actions = _suggest_ui_actions(last_user_msg, final_answer, lang)
            else:
                response_actions = []
            st.session_state.llm_messages.append({
                "role": "assistant",
                "content": final_answer,
                "actions": response_actions,
            })
        elif bg_result.get("status") == "error":
            error_msg = _handle_api_error(Exception(bg_result["answer"]), lang, render=False)
            st.session_state.llm_messages.append({
                "role": "assistant",
                "content": error_msg,
                "actions": [],
            })
        # Clear unread since panel is now open
        st.session_state["_ai_bg_unread_count"] = 0
        st.session_state["_ai_bg_response_ready"] = False

    history_container = st.container(height=history_height, border=True)
    with history_container:
        recent_messages = st.session_state.llm_messages[-8:]
        queued_prompt = st.session_state.pop("_ai_pending_question", None)

        if not recent_messages and not queued_prompt:
            _render_chat_welcome(
                lang=lang,
                panel_key=panel_key,
                history_container=history_container,
                show_starters=show_starters,
            )
        else:
            for msg_idx, msg in enumerate(recent_messages):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("actions"):
                        _render_nav_actions(msg["actions"], key_prefix=f"{panel_key}_{msg_idx}")
            if queued_prompt:
                st.info(
                    "Using page context to ask the assistant..."
                    if lang == "en" else
                    "正在带着当前页面上下文向 AI 提问..."
                )
                _submit_prompt(queued_prompt, lang, history_container, key_prefix=panel_key)

    with st.form(f"{panel_key}_form", clear_on_submit=True):
        prompt = st.text_input(
            "Ask EasyICU AI" if lang == "en" else "向 EasyICU AI 提问",
            placeholder="Ask about the current workflow..." if lang == "en" else "询问当前流程、概念或报错...",
            label_visibility="collapsed",
        )
        send_clicked = st.form_submit_button(
            "Send" if lang == "en" else "发送",
            type="primary",
            use_container_width=True,
        )
    if send_clicked and prompt.strip():
        st.session_state["_ai_pending_question"] = prompt.strip()
        st.rerun()

    if st.session_state.llm_messages:
        action_cols = st.columns(2)
        with action_cols[0]:
            st.download_button(
                "📄 Export Chat" if lang == "en" else "📄 导出对话",
                data=_build_chat_export_text(),
                file_name=f"easyicu_ai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"{panel_key}_export_chat",
            )
        with action_cols[1]:
            if st.button("🗑️ Clear Chat" if lang == "en" else "🗑️ 清空对话", key=f"{panel_key}_clear_chat", use_container_width=True):
                st.session_state.llm_messages = []
                st.session_state["_ai_pending_question"] = None
                st.rerun()


def render_floating_chat_dock():
    """Render a fixed bottom-right floating AI chat dock."""
    _init_chat_state()
    lang = st.session_state.get("language", "en")
    has_pending_prompt = bool(
        st.session_state.get("_ai_pending_question")
        or st.session_state.get("_ai_bg_unread_count", 0)
        or st.session_state.get("_ai_bg_responding", False)
    )
    if not st.session_state.get("llm_enabled", False) and not has_pending_prompt:
        return
    if "_floating_ai_open" not in st.session_state:
        st.session_state["_floating_ai_open"] = False
    if "_floating_ai_size" not in st.session_state:
        st.session_state["_floating_ai_size"] = "m"
    if st.session_state.get("_ai_pending_question"):
        st.session_state["_floating_ai_open"] = True

    size_presets = {
        "s": {
            "panel_width": "clamp(320px, 28vw, 500px)",
            "panel_max_height": "min(72vh, 680px)",
            "history_height": 300,
        },
        "m": {
            "panel_width": "clamp(360px, 34vw, 620px)",
            "panel_max_height": "min(84vh, 860px)",
            "history_height": 390,
        },
        "l": {
            "panel_width": "clamp(460px, 44vw, 860px)",
            "panel_max_height": "min(92vh, 1100px)",
            "history_height": 620,
        },
    }
    size_key = st.session_state.get("_floating_ai_size", "m")
    if size_key not in size_presets:
        size_key = "m"
        st.session_state["_floating_ai_size"] = size_key
    size_cfg = size_presets[size_key]
    export_locked = bool(st.session_state.get("_exporting_in_progress", False))

    st.markdown(
        f"""
        <style>
        :root {{
            --easyicu-ai-launcher-size: clamp(56px, 4.2vw, 78px);
            --easyicu-ai-panel-width: {size_cfg["panel_width"]};
            --easyicu-ai-panel-max-height: {size_cfg["panel_max_height"]};
            --easyicu-ai-title-size: clamp(0.92rem, 0.35vw + 0.84rem, 1.06rem);
            --easyicu-ai-subtitle-size: clamp(0.72rem, 0.18vw + 0.68rem, 0.82rem);
            --easyicu-ai-body-size: clamp(0.82rem, 0.16vw + 0.78rem, 0.93rem);
            --easyicu-ai-button-size: clamp(2rem, 2vw, 2.45rem);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
        /* ---- Animation keyframes ---- */
        @keyframes easyicuPanelSlideIn {
            from { opacity: 0; transform: translateY(24px) scale(0.96); }
            to   { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes easyicuPanelSlideOut {
            from { opacity: 1; transform: translateY(0) scale(1); }
            to   { opacity: 0; transform: translateY(24px) scale(0.96); }
        }
        @keyframes easyicuLauncherPop {
            0%   { transform: scale(0.5); opacity: 0; }
            70%  { transform: scale(1.1); }
            100% { transform: scale(1); opacity: 1; }
        }
        @keyframes easyicuBadgePulse {
            0%, 100% { transform: scale(1); }
            50%      { transform: scale(1.2); }
        }
        @keyframes easyicuBadgeBounce {
            0%   { transform: scale(0); }
            50%  { transform: scale(1.3); }
            70%  { transform: scale(0.9); }
            100% { transform: scale(1); }
        }

        div.st-key-floating_ai_launcher,
        div.st-key-floating_ai_panel {
            position: fixed !important;
            z-index: 99990 !important;
            margin: 0 !important;
        }

        div.st-key-floating_ai_launcher {
            right: clamp(12px, 1.25vw, 20px);
            bottom: clamp(12px, 1.25vw, 20px);
            width: calc(var(--easyicu-ai-launcher-size) + 12px);
            animation: easyicuLauncherPop 0.35s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        }

        /* Notification badge on the launcher button */
        div.st-key-floating_ai_launcher .ai-notif-badge {
            position: absolute;
            top: -4px; right: 2px;
            min-width: 20px; height: 20px;
            border-radius: 10px;
            background: #ef4444;
            color: #fff;
            font-size: 0.72rem;
            font-weight: 700;
            display: flex; align-items: center; justify-content: center;
            padding: 0 5px;
            box-shadow: 0 2px 8px rgba(239, 68, 68, 0.4);
            animation: easyicuBadgeBounce 0.45s cubic-bezier(0.34, 1.56, 0.64, 1) both;
            z-index: 99999;
            pointer-events: none;
        }
        div.st-key-floating_ai_launcher .ai-notif-badge.pulse {
            animation: easyicuBadgePulse 1.5s ease-in-out infinite;
        }

        div.st-key-floating_ai_launcher .stButton > button {
            width: var(--easyicu-ai-launcher-size);
            min-width: var(--easyicu-ai-launcher-size);
            height: var(--easyicu-ai-launcher-size);
            border-radius: 999px;
            border: 1px solid rgba(251, 146, 60, 0.32);
            background: linear-gradient(135deg, #fb923c 0%, #f97316 48%, #2563eb 100%);
            color: #ffffff;
            font-size: clamp(1.12rem, 1vw + 0.88rem, 1.55rem);
            box-shadow: 0 18px 44px rgba(249, 115, 22, 0.28);
        }


        div.st-key-floating_ai_panel {
            right: clamp(12px, 1.25vw, 20px);
            bottom: clamp(12px, 1.25vw, 20px);
            width: min(var(--easyicu-ai-panel-width), calc(100vw - 24px));
            max-width: calc(100vw - 24px);
            max-height: var(--easyicu-ai-panel-max-height);
            border-radius: 16px;
            border: 1px solid #cbdff2;
            background:
                radial-gradient(circle at 100% 0%, rgba(251, 146, 60, 0.13), transparent 34%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.99), rgba(247, 250, 255, 0.99));
            box-shadow: 0 24px 58px rgba(15, 23, 42, 0.2);
            backdrop-filter: blur(14px);
            padding: 0.48rem 0.5rem 0.56rem 0.5rem;
            overflow-x: hidden;
            overflow-y: auto;
            animation: easyicuPanelSlideIn 0.32s cubic-bezier(0.22, 1, 0.36, 1) both;
        }

        div.st-key-floating_ai_panel .floating-ai-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.18rem 0.12rem 0.36rem 0.12rem;
        }

        div.st-key-floating_ai_panel .floating-ai-title-row {
            display: flex;
            align-items: center;
            gap: 0.55rem;
        }

        div.st-key-floating_ai_panel .floating-ai-avatar {
            width: 2.1rem;
            height: 2.1rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            background: linear-gradient(135deg, #fb923c 0%, #f97316 72%, #ea580c 100%);
            box-shadow: 0 10px 20px rgba(249, 115, 22, 0.24);
            font-size: 1.05rem;
            flex: 0 0 auto;
        }

        div.st-key-floating_ai_panel .floating-ai-title {
            font-size: var(--easyicu-ai-title-size);
            font-weight: 900;
            color: #0f172a;
            letter-spacing: -0.02em;
        }

        div.st-key-floating_ai_panel .floating-ai-subtitle {
            font-size: var(--easyicu-ai-subtitle-size);
            color: #64748b;
            margin-top: 0.04rem;
            line-height: 1.45;
        }

        div.st-key-floating_ai_panel .floating-ai-welcome {
            background:
                radial-gradient(circle at top right, rgba(251, 146, 60, 0.13), transparent 28%),
                linear-gradient(135deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
            border: 1px solid #d5e3f3;
            border-left: 4px solid #fb923c;
            border-radius: 15px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
        }

        div.st-key-floating_ai_panel .floating-ai-welcome-title {
            font-size: clamp(0.9rem, 0.24vw + 0.84rem, 1rem);
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.3rem;
        }

        div.st-key-floating_ai_panel .floating-ai-welcome-subtitle {
            font-size: var(--easyicu-ai-body-size);
            line-height: 1.6;
            color: #334155;
            margin-bottom: 0.55rem;
        }

        div.st-key-floating_ai_panel .floating-ai-sample {
            display: grid;
            gap: 0.52rem;
            margin: 0.58rem 0 0.72rem;
        }

        div.st-key-floating_ai_panel .floating-ai-user-bubble {
            justify-self: end;
            max-width: 88%;
            color: #0f172a;
            background: #eaf3ff;
            border: 1px solid #bfdbfe;
            border-radius: 16px 16px 4px 16px;
            padding: 0.56rem 0.72rem;
            font-size: var(--easyicu-ai-body-size);
            line-height: 1.45;
            box-shadow: 0 8px 18px rgba(37, 99, 235, 0.08);
        }

        div.st-key-floating_ai_panel .floating-ai-answer-card {
            color: #0f172a;
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 16px 16px 16px 4px;
            padding: 0.62rem 0.76rem;
            font-size: var(--easyicu-ai-body-size);
            line-height: 1.55;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.045);
        }

        div.st-key-floating_ai_panel .floating-ai-recommendation {
            display: grid;
            gap: 0.18rem;
            color: #1e3a8a;
            background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 14px;
            padding: 0.58rem 0.7rem;
            font-size: clamp(0.72rem, 0.12vw + 0.69rem, 0.8rem);
        }

        div.st-key-floating_ai_panel .floating-ai-recommendation span {
            color: #64748b;
            font-size: 0.66rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 0.09em;
        }

        div.st-key-floating_ai_panel .floating-ai-recommendation strong {
            color: #0f172a;
            font-weight: 800;
            line-height: 1.42;
        }

        div.st-key-floating_ai_panel .floating-ai-welcome-hint {
            font-size: clamp(0.68rem, 0.12vw + 0.65rem, 0.76rem);
            color: #2563eb;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }

        div.st-key-floating_ai_panel div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.2rem 0.3rem 0.4rem 0.3rem;
        }

        div.st-key-floating_ai_panel [data-testid="stChatMessage"] {
            margin-bottom: 0.45rem;
        }

        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] {
            border-radius: 18px;
            padding: 0.78rem 0.95rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
            border: 1px solid rgba(148, 163, 184, 0.16);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        }

        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] p,
        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] li {
            font-size: var(--easyicu-ai-body-size);
            line-height: 1.64;
            color: #0f172a;
        }

        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] ul,
        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] ol {
            padding-left: 1.15rem;
            margin-top: 0.45rem;
            margin-bottom: 0.2rem;
        }

        div.st-key-floating_ai_panel .stButton > button {
            border-radius: 14px;
            min-height: var(--easyicu-ai-button-size);
            font-size: clamp(0.8rem, 0.12vw + 0.77rem, 0.9rem);
            padding-top: 0.38rem;
            padding-bottom: 0.38rem;
            border-color: #cbdff2 !important;
            color: #1f3b63 !important;
            background: linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%) !important;
        }

        div.st-key-floating_ai_panel .stButton > button[kind="primary"],
        div.st-key-floating_ai_panel .stButton > button[data-testid="stBaseButton-primary"],
        div.st-key-floating_ai_panel form button,
        div.st-key-floating_ai_panel div[data-testid="stFormSubmitButton"] button,
        div.st-key-floating_ai_panel .stButton > button:hover {
            color: #ffffff !important;
            border-color: #1d7ef2 !important;
            background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%) !important;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.18);
        }

        div.st-key-floating_ai_panel .stButton > button[kind="primary"] *,
        div.st-key-floating_ai_panel .stButton > button[data-testid="stBaseButton-primary"] *,
        div.st-key-floating_ai_panel form button *,
        div.st-key-floating_ai_panel div[data-testid="stFormSubmitButton"] button *,
        div.st-key-floating_ai_panel .stButton > button:hover * {
            color: #ffffff !important;
            fill: #ffffff !important;
        }

        div.st-key-floating_ai_launcher .stButton > button:hover {
            filter: brightness(1.02);
        }

        div.st-key-floating_ai_panel .stChatInput {
            margin-top: 0.3rem;
        }

        div.st-key-floating_ai_panel input,
        div.st-key-floating_ai_panel textarea,
        div.st-key-floating_ai_panel label,
        div.st-key-floating_ai_panel [data-testid="stMarkdownContainer"] p,
        div.st-key-floating_ai_panel [data-testid="stCaptionContainer"] {
            font-size: var(--easyicu-ai-body-size) !important;
        }

        @media (max-width: 1512px) {
            :root {
                --easyicu-ai-panel-width: clamp(330px, 31vw, 540px);
                --easyicu-ai-panel-max-height: min(80vh, 740px);
                --easyicu-ai-title-size: clamp(0.88rem, 0.26vw + 0.82rem, 0.98rem);
                --easyicu-ai-subtitle-size: clamp(0.68rem, 0.14vw + 0.64rem, 0.76rem);
                --easyicu-ai-body-size: clamp(0.78rem, 0.12vw + 0.75rem, 0.86rem);
                --easyicu-ai-button-size: clamp(1.86rem, 1.7vw, 2.2rem);
            }

            div.st-key-floating_ai_panel {
                border-radius: 16px;
                padding: 0.28rem 0.28rem 0.34rem 0.28rem;
            }

            div.st-key-floating_ai_panel .floating-ai-welcome {
                padding: 0.8rem 0.86rem;
                margin-bottom: 0.6rem;
            }
        }

        @media (max-width: 1280px) {
            :root {
                --easyicu-ai-panel-width: clamp(320px, 29vw, 500px);
                --easyicu-ai-panel-max-height: min(78vh, 680px);
                --easyicu-ai-body-size: 0.8rem;
            }
        }

        @media (max-width: 768px) {
            div.st-key-floating_ai_launcher {
                right: 12px;
                bottom: 12px;
            }

            div.st-key-floating_ai_panel {
                left: 12px;
                right: 12px;
                bottom: 82px;
                width: auto;
                max-height: min(74vh, 640px);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Check for unread background responses
    _unread = st.session_state.get("_ai_bg_unread_count", 0)
    _responding = st.session_state.get("_ai_bg_responding", False)
    if not st.session_state.get("_floating_ai_open", False):
        with st.container(key="floating_ai_launcher"):
            if _unread > 0:
                _badge_class = "ai-notif-badge pulse" if _responding else "ai-notif-badge"
                st.markdown(
                    f'<div class="{_badge_class}">{_unread}</div>',
                    unsafe_allow_html=True,
                )
            elif _responding:
                st.markdown(
                    '<div class="ai-notif-badge pulse">⋯</div>',
                    unsafe_allow_html=True,
                )
            if st.button("💬", key="_floating_ai_open_btn", help="Open AI chat" if lang == "en" else "打开 AI 对话"):
                st.session_state["_floating_ai_open"] = True
                st.rerun()

    if st.session_state.get("_floating_ai_open", False):
        with st.container(key="floating_ai_panel"):
            dock_title = "AI Assistant" if lang == "en" else "AI 助手"
            dock_subtitle = (
                "Grounded in your current page and selections."
                if lang == "en" else
                "基于当前页面和已选配置给出建议。"
            )
            header_cols = st.columns([4.8, 0.8, 0.8, 0.8, 1, 1])
            with header_cols[0]:
                st.markdown(
                    f'''
                    <div class="floating-ai-header">
                        <div class="floating-ai-title-row">
                            <span class="floating-ai-avatar">🤖</span>
                            <div>
                                <div class="floating-ai-title">{dock_title}</div>
                                <div class="floating-ai-subtitle">{dock_subtitle}</div>
                            </div>
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
            with header_cols[1]:
                if st.button("S", key="_floating_ai_size_s_btn", use_container_width=True, help="Compact size" if lang == "en" else "紧凑尺寸", disabled=export_locked):
                    st.session_state["_floating_ai_size"] = "s"
                    st.rerun()
            with header_cols[2]:
                if st.button("M", key="_floating_ai_size_m_btn", use_container_width=True, help="Medium size" if lang == "en" else "中等尺寸", disabled=export_locked):
                    st.session_state["_floating_ai_size"] = "m"
                    st.rerun()
            with header_cols[3]:
                if st.button("L", key="_floating_ai_size_l_btn", use_container_width=True, help="Large size" if lang == "en" else "大尺寸", disabled=export_locked):
                    st.session_state["_floating_ai_size"] = "l"
                    st.rerun()
            with header_cols[4]:
                if st.button("—", key="_floating_ai_minimize_btn", use_container_width=True, help="Minimize" if lang == "en" else "最小化", disabled=export_locked):
                    _close_floating_ai_panel(disable_assistant=False)
                    st.rerun()
            with header_cols[5]:
                if st.button("✕", key="_floating_ai_close_btn", use_container_width=True, help="Close" if lang == "en" else "关闭", disabled=export_locked):
                    _close_floating_ai_panel(disable_assistant=True)
                    st.rerun()

            if not st.session_state.llm_enabled:
                st.caption(
                    "Enable AI Assistant in the sidebar settings first."
                    if lang == "en" else
                    "请先在侧边栏上方开启 AI 助手。"
                )
            elif not _is_configured():
                st.caption(
                    "Configure provider/API key in the sidebar settings first."
                    if lang == "en" else
                    "请先在侧边栏上方配置服务商/API Key。"
                )
            else:
                _render_compact_chat_panel(
                    lang=lang,
                    panel_key="_llm_floating",
                    history_height=size_cfg["history_height"],
                    show_starters=False,
                )



# ---------------------------------------------------------------------------
# Intro / tips helpers
# ---------------------------------------------------------------------------

def _render_intro(lang: str):
    """Show a brief feature overview when the assistant is disabled."""
    if lang == "en":
        st.markdown("""\
#### What is the AI Assistant?
A built-in conversational helper that knows EasyICU inside-out. It can:
- 🎯 Start from your study goal, then map it to the right EasyICU workflow
- 📖 Explain which cohort filters, feature modules, and scores fit your task
- 🗄️ List supported databases, concepts, and scoring systems
- 📊 Help interpret extraction results (SOFA, Sepsis-3, missingness, etc.)

**Getting started:**
1. Toggle **Enable AI Assistant** in the sidebar
2. Choose a provider and enter your API key or token
3. Start by describing your task, e.g. "I want to build a sepsis early-warning cohort."
4. If you already know the disease cohort you want, ask directly for AKI / Sepsis / ventilation / ICD-based filtering.
""")
    else:
        st.markdown("""\
#### AI 助手是什么？
内置的对话助手，熟知 EasyICU 的所有功能。它可以：
- 🎯 从你的研究目标出发，反推最合适的 EasyICU 工作流
- 📖 解释适合该任务的队列筛选、特征模块和临床评分
- 🗄️ 列出支持的数据库、概念和评分系统
- 📊 帮助解读提取结果（SOFA、Sepsis-3、缺失率等）

**快速开始：**
1. 在侧边栏开启 **启用 AI 助手**
2. 选择服务商并填写对应的 API Key / Token
3. 先描述你的任务，例如“我想做脓毒症实时预警队列”。
4. 如果你已经知道要筛选的疾病队列，也可以直接问 AKI / Sepsis / 机械通气 / ICD 队列怎么设置。
""")


def _render_tips(lang: str):
    if lang == "en":
        st.markdown("""\
- **Onboarding**: "I want to extract SOFA-2 from MIMIC-IV. What exact steps should I follow in the web UI?"
- **Task-first planning**: "I want to build a sepsis early-warning model. How can EasyICU support this, and which modules should I extract?"
- **Disease cohort setup**: "I want to build an AKI cohort. Which cohort filters should I enable, and which concepts should I export?"
- **Trajectory analysis**: "I want to cluster sepsis trajectories over time. Which time-series concepts and scores should I export?"
- **Feature planning**: "For septic shock research, which EasyICU concepts should I export besides SOFA and vasopressors?"
- **ICD filtering**: "I want an ICD-defined pneumonia or heart-failure cohort in MIMIC-IV. How should I use the ICD filter in Step 2?"
- **Cross-database mapping**: "Which respiratory concepts are available across miiv, mimic, eicu, aumc, hirid, and sic?"
- **Troubleshooting**: "My exported data shows high missingness for fio2. What are the most likely causes and checks?"
- **Interpretation**: "How should I interpret `sep3_sofa2`, `susp_inf`, and `sofa2` together?"
- **Definition help**: "Explain the Sepsis suspected-infection settings in the web app and when I should use `auto`, `and`, or `icd_abx`."
- **Code-aware help**: "Where is export implemented in app.py?" / "How does `load_concepts` work?"
- **Evidence-backed answers**: "Explain Sepsis-3 with PubMed sources and relate it to EasyICU outputs."
- **Python workflow**: "Show me a minimal Python example to load pafi, sofa2, and sep3_sofa2."
""")
    else:
        st.markdown("""\
- **新手引导**: "我想从 MIMIC-IV 提取 SOFA-2，网页端具体点哪里、按什么顺序做？"
- **任务规划**: "我想做脓毒症实时预警，EasyICU 能怎么支持，建议提取哪些模块？"
- **疾病队列设置**: "我想构建 AKI 队列，Step 2 应该怎么筛，Step 3 建议提取哪些特征？"
- **轨迹分析**: "我想做脓毒症患者轨迹聚类，应该优先导出哪些时间序列特征和评分？"
- **选特征建议**: "如果我要做脓毒症休克研究，除了 SOFA 和升压药，还建议导出哪些概念？"
- **ICD 队列**: "我想在 MIMIC-IV 里按 ICD 筛肺炎或心衰患者，Step 2 应该怎么设置？"
- **跨库对照**: "miiv、mimic、eicu、aumc、hirid、sic 里哪些呼吸相关概念都能取到？"
- **排错诊断**: "我导出的 fio2 缺失率很高，最可能是什么原因，应该检查哪几步？"
- **结果解读**: "`sep3_sofa2`、`susp_inf` 和 `sofa2` 应该怎么一起解释？"
- **定义说明**: "请解释网页端的 Sepsis 疑似感染设置，什么时候该用 `auto`、`and` 或 `icd_abx`？"
- **代码问题**: "app.py 里 export 在哪实现？" / "`load_concepts` 是怎么工作的？"
- **带证据医学回答**: "结合 PubMed 解释 Sepsis-3，并对应到 EasyICU 的输出概念。"
- **Python 工作流**: "给我一个最小 Python 例子，同时加载 pafi、sofa2 和 sep3_sofa2。"
""")


# ---------------------------------------------------------------------------
# Fast local replies for trivial prompts
# ---------------------------------------------------------------------------

def _normalize_prompt(text: str) -> str:
    """Normalize trivial user input for lightweight local handling."""
    text = (text or "").strip().lower()
    text = re.sub(r"[!！?？,.，。~～\s]+", "", text)
    return text


def _get_instant_reply(prompt: str, lang: str) -> str | None:
    """Short-circuit greetings so the first turn feels immediate."""
    normalized = _normalize_prompt(prompt)
    if not normalized:
        return None

    greetings = {
        "hi", "hello", "hey", "yo", "hiya",
        "你好", "您好", "哈喽", "嗨", "在吗", "在嘛",
    }
    if normalized in greetings:
        return (
            "你好，我是 EasyICU 助手。可以直接问我 EasyICU 的功能、概念、评分或数据流程。"
            if lang != "en" else
            "Hi, I'm the EasyICU assistant. Ask me about features, concepts, scores, or workflow steps."
        )

    code_access = {
        "你能看项目代码吗", "你能看代码吗", "能看项目代码吗", "能看代码吗",
        "canyouseetheprojectcode", "canyouseethecode", "canyoureadthecode",
    }
    if normalized in code_access:
        return (
            "可以。这个助手现在会结合 EasyICU 本地代码摘要来回答实现问题。你可以直接问文件、函数或流程，比如 `app.py 里 export 是怎么做的`。"
            if lang != "en" else
            "Yes. This assistant can answer against a local EasyICU code snapshot. Ask about files, functions, or flows such as `how export works in app.py`."
        )

    return None


# ---------------------------------------------------------------------------
# Background LLM response (when panel is minimized)
# ---------------------------------------------------------------------------

import threading

_bg_lock = threading.Lock()


def _bg_llm_call(messages: list, lang: str, provider: str, model: str,
                 base_url: str, api_key: str, session_id: str):
    """Run an LLM call in a background thread and store the result."""
    try:
        import openai
        client_kwargs = {"base_url": base_url}
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            client_kwargs["api_key"] = "unused"
        # Strip proxy env vars
        env_backup = {}
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                    "ALL_PROXY", "all_proxy"):
            if var in os.environ:
                env_backup[var] = os.environ.pop(var)
        try:
            client = openai.OpenAI(**client_kwargs)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
            )
            draft = _strip_llm_reasoning(resp.choices[0].message.content or "")
        finally:
            os.environ.update(env_backup)

        # Store result in a module-level dict (accessible cross-rerun)
        with _bg_lock:
            _bg_results[session_id] = {
                "status": "done",
                "answer": draft,
                "lang": lang,
            }
    except Exception as exc:
        with _bg_lock:
            _bg_results[session_id] = {
                "status": "error",
                "answer": str(exc),
                "lang": lang,
            }


# Module-level dict to store background results (survives Streamlit reruns)
_bg_results: dict[str, dict] = {}


def _start_bg_response(prompt: str, lang: str) -> str | None:
    """Prepare and start a background LLM call. Returns a session_id or None."""
    if not _is_configured():
        return None
    try:
        messages, tool_events = _compose_agent_messages(prompt)
    except Exception:
        return None
    st.session_state.llm_last_tool_events = tool_events
    provider = coerce_public_provider(st.session_state.get("llm_provider", public_default_provider_key()))
    p_info = public_provider_defaults(provider)
    model = (st.session_state.get("llm_model", "").strip() or p_info[2])
    base_url = st.session_state.get("llm_base_url", "").strip() or p_info[1]
    api_key = st.session_state.get("llm_api_key", "")
    session_id = f"bg_{id(st.session_state)}_{len(st.session_state.llm_messages)}"
    t = threading.Thread(
        target=_bg_llm_call,
        args=(messages, lang, provider, model, base_url, api_key, session_id),
        daemon=True,
    )
    t.start()
    return session_id


def _check_bg_response() -> dict | None:
    """Check if any background response is ready. Returns result dict or None."""
    session_id = st.session_state.get("_ai_bg_session_id")
    if not session_id:
        return None
    with _bg_lock:
        result = _bg_results.pop(session_id, None)
    return result


# ---------------------------------------------------------------------------
# Streaming & error handling
# ---------------------------------------------------------------------------

def _stream_response(messages: list, lang: str):
    """Call the LLM API with streaming and render tokens incrementally."""
    client = _get_client()
    if client is None:
        err = ("Failed to create API client. Check your configuration."
               if lang == "en" else "无法创建 API 客户端，请检查配置。")
        st.error(err)
        return

    provider = coerce_public_provider(st.session_state.get("llm_provider", public_default_provider_key()))
    model = (st.session_state.get("llm_model", "").strip()
             or public_provider_defaults(provider)[2])
    if not model:
        st.error("⚠️ " + ("No model specified." if lang == "en" else "未指定模型名称。"))
        return

    status_placeholder = st.empty()
    answer_placeholder = st.empty()

    try:
        status_placeholder.info(
            "🤔 Thinking..." if lang == "en" else "🤔 正在思考..."
        )
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        status_placeholder.info(
            "✍️ Generating response..." if lang == "en" else "✍️ 正在生成回答..."
        )
        final_response = _stream_text(stream, answer_placeholder)
        final_response = _append_quick_links(
            prompt=st.session_state.llm_messages[-1]["content"],
            answer=final_response,
            lang=lang,
        )
        answer_placeholder.markdown(final_response)
        st.session_state.llm_last_verification = None
        response_actions = _suggest_ui_actions(
            st.session_state.llm_messages[-1]["content"],
            final_response,
            lang,
        )
        _render_nav_actions(response_actions, key_prefix="_llm_action_live")
        status_placeholder.empty()
        st.session_state.llm_messages.append(
            {
                "role": "assistant",
                "content": final_response,
                "actions": response_actions,
            }
        )
    except Exception as exc:
        status_placeholder.empty()
        error_message = _handle_api_error(exc, lang, render=False)
        answer_placeholder.markdown(error_message)
        st.session_state.llm_messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "actions": [],
            }
        )


def _token_generator(stream):
    """Yield content tokens from an OpenAI-style streaming response."""
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                continue
            token = getattr(delta, "content", None)
            if token:
                yield token


def _handle_api_error(exc: Exception, lang: str, render: bool = True) -> str:
    """Build and optionally display a user-friendly error for common API failures."""
    err_str = str(exc)
    err_lower = err_str.lower()
    if "authentication" in err_str.lower() or "401" in err_str:
        provider = st.session_state.get("llm_provider", "custom")
        if provider == "huggingface_free":
            msg = (
                "❌ Hugging Face requires your own token. "
                "Please create an HF token and paste it here."
                if lang == "en" else
                "❌ Hugging Face 需要你自己提供 token。"
                "请创建 HF token 后再填写。"
            )
        else:
            msg = ("❌ Authentication failed — please check your API Key."
                   if lang == "en" else "❌ 认证失败 — 请检查 API Key 是否正确。")
    elif (
        "429" in err_str
        or "rate" in err_lower
        or "temporarily rate-limited" in err_lower
        or "provider returned error" in err_lower
        or "retry shortly" in err_lower
    ):
        msg = (
            "⏳ The current hosted model is being rate-limited upstream. Please retry shortly, or switch the hosted default model to a more stable free model."
            if lang == "en" else
            "⏳ 当前托管模型被上游限流了。请稍后重试，或把 hosted 默认模型切换到更稳定的免费模型。"
        )
    elif "socksio" in err_lower or "using socks proxy" in err_lower:
        msg = (
            "🌐 Proxy configuration error — the app detected a SOCKS proxy from the environment. "
            "EasyICU now ignores system proxy variables by default. Please retry. "
            "If you explicitly need a SOCKS proxy, install `httpx[socks]`."
            if lang == "en" else
            "🌐 代理配置异常 — 应用检测到了环境变量中的 SOCKS 代理。"
            "EasyICU 现在默认忽略系统代理变量，请重试。"
            "如果你确实需要 SOCKS 代理，请安装 `httpx[socks]`。"
        )
    elif "model" in err_lower or "404" in err_str:
        msg = ("⚠️ Model not found — please verify the model name."
               if lang == "en" else "⚠️ 模型未找到 — 请确认模型名称是否正确。")
    elif "connect" in err_lower or "timeout" in err_lower:
        msg = ("🌐 Connection error — check the API Base URL and your network."
               if lang == "en" else "🌐 连接失败 — 请检查 API Base URL 和网络连接。")
    else:
        msg = ("❌ API error: " if lang == "en" else "❌ API 调用出错: ") + err_str
    if render:
        st.error(msg)
    return msg
