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
import html
import os
import re
from collections.abc import MutableMapping
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
from easyicu.webapp.session_state import clear_agent_continuation_state

_FEATURE_COUNT = len(get_all_concepts())

COPILOT_STUDY_STEPS: tuple[tuple[str, str], ...] = (
    ("question", "Research question"),
    ("data", "Data source"),
    ("cohort", "Cohort"),
    ("concepts", "Feature modules"),
    ("extract", "Extraction"),
    ("review", "Review"),
    ("analysis", "Analysis run"),
    ("draft", "Draft gate"),
)
COPILOT_STEP_INDEX = {step: idx for idx, (step, _label) in enumerate(COPILOT_STUDY_STEPS)}
COPILOT_DEFAULT_MODULES = [
    "Demographics",
    "Vital signs",
    "Labs",
    "SOFA / SOFA-2",
    "Sepsis-3",
    "Outcomes",
]
COPILOT_BRANCH_CONFIG = {
    "predict": {
        "chip": "Predict sepsis mortality",
        "question_en": "Among Sepsis-3 patients, do first-24h bedside features predict in-hospital mortality, and does adding lactate improve it?",
        "question_zh": "在 Sepsis-3 患者中，前 24 小时床旁特征能否预测院内死亡，加入乳酸是否改善模型？",
        "review_target": "quick_viz",
        "selected_concepts": ["hr", "map", "temp", "spo2", "sofa2", "lact", "age", "death"],
        "why": {
            "question": "A vague aim becomes testable only after binding cohort, outcome, time window, and comparator.",
            "data": "Demo data lowers risk for the first pass; real data stays local and uses the same gates.",
            "cohort": "The cohort defines every downstream denominator, rate, and model row.",
            "concepts": "Only question-relevant modules are preselected, then coverage is audited before modelling.",
            "extract": "A frozen normalized frame keeps review panels and agent runs reproducible.",
            "review": "A human preview catches obvious data problems before spending an analysis run.",
            "analysis": "The run is evidence-bound; every table or figure must trace to an artifact.",
            "draft": "Drafting stays locked until checks pass and a human signs off.",
        },
    },
    "crossdb": {
        "chip": "Compare across ICU databases",
        "question_en": "Does the sepsis mortality signal replicate across ICU databases, and where do feature distributions diverge?",
        "question_zh": "脓毒症死亡信号能否跨 ICU 数据库复现，哪些特征分布差异最大？",
        "review_target": "cross_db",
        "selected_concepts": ["hr", "map", "lact", "sofa2", "death", "age"],
        "why": {
            "question": "Replication needs one shared cohort definition applied consistently across databases.",
            "data": "Cross-database work needs at least two local sources; demo mode seeds all six safely.",
            "cohort": "Here the cohort includes both patients and the database set being compared.",
            "concepts": "Only concepts available across selected databases can support fair comparisons.",
            "extract": "Each source is normalized into the same concept names before comparison.",
            "review": "Availability and distribution review prevents over-reading database differences.",
            "analysis": "Per-database summaries and deltas are logged before any replication claim.",
            "draft": "Cross-database claims remain gated until each database artifact is traceable.",
        },
    },
    "quality": {
        "chip": "Audit data quality first",
        "question_en": "Before modelling, which ICU concepts are sparse, out-of-range, or trustworthy enough to analyze?",
        "question_zh": "建模前，哪些 ICU 概念稀疏、越界，哪些足够可信可以进入分析？",
        "review_target": "cohort",
        "selected_concepts": ["hr", "map", "temp", "spo2", "crea", "lact", "urine", "sofa2", "death"],
        "why": {
            "question": "A quality-first study starts with trustworthiness instead of effect estimates.",
            "data": "The same extraction gates apply; the first deliverable is a coverage and range audit.",
            "cohort": "The audit must use the same denominator as the later analysis would use.",
            "concepts": "Broad modules reveal sparse or unsafe variables before they bias a model.",
            "extract": "Consistent frames let coverage, ranges, and temporal density be measured uniformly.",
            "review": "The review table is the main deliverable for a QC-first branch.",
            "analysis": "Coverage, range, missingness, and density checks produce flags, not claims.",
            "draft": "Even a QC summary must trace every flag to logged evidence.",
        },
    },
}

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
- **Research Copilot** — chat-first help for navigation, feature planning, troubleshooting, evidence lookup, and handoff into the auditable Research Agent

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


def _default_copilot_study_state(state: MutableMapping[str, object] | None = None) -> dict[str, object]:
    state = state or {}
    patient_n = int(state.get("demo_mode_patients") or 10)
    return {
        "branch": None,
        "step": "question",
        "data_mode": state.get("entry_mode") if state.get("entry_mode") in {"demo", "real"} else "demo",
        "patient_n": patient_n,
        "db_count": 6,
        "outcome": "In-hospital mortality",
        "window": "first 24h",
        "exposure": "lactate",
        "modules": COPILOT_DEFAULT_MODULES[:],
        "question": "",
        "draft_signed": False,
        "last_update": datetime.now().isoformat(timespec="seconds"),
    }


def _ensure_copilot_study_state(state: MutableMapping[str, object]) -> dict[str, object]:
    study = state.get("_copilot_guided_study")
    if not isinstance(study, dict):
        study = _default_copilot_study_state(state)
        state["_copilot_guided_study"] = study
    for key, value in _default_copilot_study_state(state).items():
        study.setdefault(key, value)
    return study


def _reset_copilot_study_state(state: MutableMapping[str, object]) -> dict[str, object]:
    study = _default_copilot_study_state(state)
    state["_copilot_guided_study"] = study
    return study


def _copilot_pick_branch(text: str) -> str:
    text_l = (text or "").lower()
    if any(key in text_l for key in ("cross", "database", "databases", "replicate", "replication", "多库", "跨库", "数据库")):
        return "crossdb"
    if any(key in text_l for key in ("quality", "missing", "coverage", "audit", "sparse", "trust", "qc", "缺失", "质量", "覆盖")):
        return "quality"
    return "predict"


def _copilot_endpoint_pinned(text: str) -> bool:
    text_l = (text or "").lower()
    return bool(re.search(r"in-?hospital|28[\s-]*day|icu\s+mortality|icu\s+death|院内|28\s*天|icu\s*死亡", text_l))


def _copilot_apply_entities(study: MutableMapping[str, object], text: str) -> list[str]:
    text_l = (text or "").lower()
    found: list[str] = []
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
    patient_match = re.search(r"\b(\d{1,3})\s*(?:patient|patients|case|cases|stay|stays|subject|subjects|例|人)", text_l)
    if patient_match:
        patient_n = max(5, min(50, int(patient_match.group(1))))
        study["patient_n"] = patient_n
        found.append(f"{patient_n} stays")
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    return found


def _copilot_frame_question(study: MutableMapping[str, object], lang: str) -> str:
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])
    if branch == "predict":
        if lang == "en":
            return (
                f"Among Sepsis-3 patients, do {study.get('window', 'first 24h')} bedside features "
                f"predict {str(study.get('outcome', 'In-hospital mortality')).lower()}, and does adding "
                f"{study.get('exposure', 'lactate')} improve the model?"
            )
        return (
            f"在 Sepsis-3 患者中，{study.get('window', '前 24 小时')}床旁特征能否预测"
            f"{study.get('outcome', '院内死亡')}，加入 {study.get('exposure', '乳酸')} 是否改善模型？"
        )
    return str(config["question_en"] if lang == "en" else config["question_zh"])


def _copilot_status_markdown(study: MutableMapping[str, object], lang: str) -> str:
    active_step = str(study.get("step") or "question")
    active_idx = COPILOT_STEP_INDEX.get(active_step, 0)
    rows = []
    for idx, (step, label_en) in enumerate(COPILOT_STUDY_STEPS):
        mark = "[x]" if idx < active_idx or (step == "draft" and study.get("draft_signed")) else ("[>]" if idx == active_idx else "[ ]")
        label = label_en if lang == "en" else {
            "question": "研究问题",
            "data": "数据源",
            "cohort": "队列",
            "concepts": "特征模块",
            "extract": "提取",
            "review": "审阅",
            "analysis": "分析运行",
            "draft": "草稿闸门",
        }.get(step, label_en)
        rows.append(f"{mark} {label}")
    return "\n".join(rows)


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

    if step in {"data", "cohort", "concepts", "extract"}:
        workflow("workflow_study_extract", "Open Classic Flow", "打开经典流程", "study_extract")
    if step in {"review", "analysis", "draft"}:
        workflow("workflow_study_review", "Open Review Workspace", "打开审阅工作区", "study_review")
    if step in {"analysis", "draft"}:
        actions.append({
            "id": "agent_handoff",
            "kind": "agent_handoff",
            "label": "Hand off to Research Agent" if is_en else "交给 Research Agent",
        })
    if str(study.get("data_mode")) == "real":
        workflow("workflow_real_extraction", "Open Real Data Setup", "打开真实数据配置", "real_extraction")
    return actions[:3]


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
    status_title = "Study workspace" if lang == "en" else "研究工作区"
    return f"{body}\n\n**{status_title}**\n\n```text\n{_copilot_status_markdown(study, lang)}\n```"


def _copilot_advance_step(study: MutableMapping[str, object]) -> str:
    current = str(study.get("step") or "question")
    sequence = [step for step, _label in COPILOT_STUDY_STEPS]
    try:
        idx = sequence.index(current)
    except ValueError:
        idx = 0
    next_step = sequence[min(idx + 1, len(sequence) - 1)]
    study["step"] = next_step
    study["last_update"] = datetime.now().isoformat(timespec="seconds")
    return next_step


def _handle_copilot_guided_prompt(
    prompt: str,
    lang: str,
    state: MutableMapping[str, object] | None = None,
) -> tuple[str, list[dict[str, object]]] | None:
    """Handle local chat-first study control before falling back to an LLM."""
    state = state if state is not None else st.session_state
    prompt = (prompt or "").strip()
    if not prompt:
        return None
    text_l = prompt.lower()
    study = _ensure_copilot_study_state(state)
    guided_active = bool(study.get("branch"))
    guided_intent = any(key in text_l for key in (
        "run the whole", "autopilot", "do it for me", "guided study", "whole demo",
        "walk me", "start a guided", "use demo", "use local", "go back", "why this step",
        "why?", "研究", "帮我跑", "自动跑", "一键", "回退", "为什么",
    ))
    branch_intent = _copilot_pick_branch(prompt) != "predict" or any(key in text_l for key in (
        "sepsis", "mortality", "lactate", "aki", "trajectory", "cohort", "prediction",
        "脓毒症", "死亡", "乳酸", "队列", "预测",
    ))
    if not (guided_active or guided_intent or branch_intent):
        return None

    if any(key in text_l for key in ("new study", "start over", "reset study", "重新开始", "新研究")):
        study = _reset_copilot_study_state(state)
        body = (
            "New study started. Describe the question, pick a direction, or say `run the whole demo`."
            if lang == "en" else
            "已开始新研究。请描述问题、选择方向，或直接说“跑完整演示”。"
        )
        return _copilot_reply(study, body, lang), []

    if not study.get("branch"):
        study["branch"] = _copilot_pick_branch(prompt)
    _copilot_apply_entities(study, prompt)
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])

    if any(key in text_l for key in ("back", "go back", "undo", "change", "edit", "previous", "回退", "修改", "上一步")):
        current_idx = COPILOT_STEP_INDEX.get(str(study.get("step") or "question"), 0)
        editable = ["question", "data", "cohort", "concepts"]
        target = "question"
        for step in reversed(editable):
            if COPILOT_STEP_INDEX[step] < current_idx:
                target = step
                break
        study["step"] = target
        body = (
            f"Rewound to **{dict(COPILOT_STUDY_STEPS)[target]}**. Downstream choices will be refreshed from here."
            if lang == "en" else
            f"已回退到 **{target}**。后续选择会从这里重新刷新。"
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if any(key in text_l for key in ("why", "explain", "reason", "为什么", "解释")):
        step = str(study.get("step") or "question")
        why = str(config["why"].get(step, config["why"]["question"]))
        body = (
            f"Why this step: {why}"
            if lang == "en" else
            f"为什么做这一步：{why}"
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if re.search(r"\b\d{1,3}\b", text_l) and any(key in text_l for key in ("patient", "patients", "stay", "stays", "cohort", "sample", "例", "人")):
        patient_n = int(study.get("patient_n") or 10)
        body = (
            f"Set the demo cohort to **{patient_n} stays**. I will keep downstream review and analysis tied to that denominator."
            if lang == "en" else
            f"已把演示队列设为 **{patient_n} 例 ICU stay**。后续审阅和分析会绑定这个分母。"
        )
        if COPILOT_STEP_INDEX.get(str(study.get("step") or "question"), 0) > COPILOT_STEP_INDEX["cohort"]:
            study["step"] = "cohort"
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if any(key in text_l for key in ("run the whole", "whole demo", "autopilot", "just do it", "do it for me", "帮我跑", "自动跑", "一键")):
        study["data_mode"] = "demo"
        study["step"] = "draft"
        study["draft_signed"] = False
        study["question"] = _copilot_frame_question(study, lang)
        state["_copilot_autopilot_ready"] = True
        body = (
            "I ran the guided demo path to the evidence gate: framed the question, selected demo data, built the cohort, chose modules, prepared extraction, loaded review, and assembled the analysis run. The draft remains locked until you review evidence and sign off."
            if lang == "en" else
            "我已把引导式演示推进到证据闸门：完成问题框定、演示数据、队列、模块、提取准备、审阅加载和分析运行组装。草稿仍会锁定，直到你审阅证据并确认。"
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if study.get("step") == "question" and branch == "predict" and not _copilot_endpoint_pinned(prompt):
        study["step"] = "question"
        body = (
            "Good direction. Quick check before I build the plan: do you mean **in-hospital mortality**, **28-day mortality**, or **ICU mortality**?"
            if lang == "en" else
            "方向可以。生成计划前先确认：你指的是 **院内死亡**、**28 天死亡**，还是 **ICU 死亡**？"
        )
        return _copilot_reply(study, body, lang), []

    if any(key in text_l for key in ("28-day", "28 day", "28天", "icu mortality", "icu death", "in-hospital", "院内")) and study.get("step") == "question":
        study["question"] = _copilot_frame_question(study, lang)
        study["step"] = "data"
        body = (
            f"Got it. I framed the study as: **{study['question']}**\n\nNext, choose how data enters: demo data for a fast walkthrough, or local real data."
            if lang == "en" else
            f"收到。我把研究问题框定为：**{study['question']}**\n\n下一步选择数据入口：快速演示数据，或本地真实数据。"
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if any(key in text_l for key in ("real data", "local data", "use local", "真实数据", "本地数据")):
        study["data_mode"] = "real"
        study["step"] = "data"
        body = (
            "Real-data mode selected. I can open the classic data-source setup so you can choose the database and local path; patient rows stay on this machine."
            if lang == "en" else
            "已选择真实数据模式。我可以打开经典数据源配置页，让你选择数据库和本地路径；患者行数据不会离开本机。"
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if any(key in text_l for key in ("demo", "use demo", "演示", "示例")) and study.get("step") in {"data", "question"}:
        study["data_mode"] = "demo"
        study["step"] = "cohort"
        study["question"] = _copilot_frame_question(study, lang)
        body = (
            "Demo data selected. I set up a lightweight, reproducible cohort so you can inspect the workflow without tokens or uploads."
            if lang == "en" else
            "已选择演示数据。我会使用轻量、可复现的队列，让你不用 token、也不用上传数据就能检查流程。"
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    if study.get("step") == "question":
        study["question"] = _copilot_frame_question(study, lang)
        study["step"] = "data"
        body = (
            f"I framed your study as: **{study['question']}**\n\nNext, choose data source: say `use demo` for a safe walkthrough or `use local data` for real extraction."
            if lang == "en" else
            f"我把你的研究问题框定为：**{study['question']}**\n\n下一步选择数据源：说“用演示数据”可快速体验，说“用本地数据”则进入真实提取。"
        )
        return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)

    next_step = _copilot_advance_step(study)
    if next_step == "cohort":
        body = (
            f"Cohort ready: **{study.get('patient_n', 10)} demo stays** with the branch `{config['chip']}`. You can say `use 30 patients` to edit the denominator."
            if lang == "en" else
            f"队列已准备：**{study.get('patient_n', 10)} 例演示 ICU stay**，研究方向为 `{config['chip']}`。你可以说“用 30 个患者”来修改分母。"
        )
    elif next_step == "concepts":
        body = (
            "I preselected the modules this question needs: demographics, vitals, labs, SOFA/SOFA-2, Sepsis-3, and outcomes. Coverage will be audited before analysis."
            if lang == "en" else
            "我已预选该问题需要的模块：人口学、生命体征、实验室、SOFA/SOFA-2、Sepsis-3 和结局。分析前会先做覆盖率审计。"
        )
    elif next_step == "extract":
        body = (
            "Extraction is ready. I can open the classic 4-step flow now, or continue in chat and then load Patient Review."
            if lang == "en" else
            "提取已准备好。我可以现在打开经典四步流程，也可以继续在聊天中推进后再加载患者审阅。"
        )
    elif next_step == "review":
        body = (
            "Review is ready. Open Patient Review to inspect data tables, time series, patient overview, and quality flags."
            if lang == "en" else
            "审阅已准备好。可以打开患者审阅页查看表格、时间序列、患者概览和质量标记。"
        )
    elif next_step == "analysis":
        body = (
            "Analysis run assembled. It remains evidence-bound: deterministic steps must complete and produce traceable artifacts before any draft claim unlocks."
            if lang == "en" else
            "分析运行已组装。它仍然保持证据绑定：确定性步骤必须完成并产生可追溯产物，草稿论断才会解锁。"
        )
    else:
        study["draft_signed"] = False
        body = (
            "The draft gate is reached but locked. Review the workspace or hand the framed question to Research Agent; no claim is written until evidence checks pass and you sign off."
            if lang == "en" else
            "已到达草稿闸门，但仍处于锁定状态。请审阅工作区或把问题交给 Research Agent；证据检查通过并人工确认前不会写出论断。"
        )
    return _copilot_reply(study, body, lang), _copilot_study_actions(study, lang)


def _local_copilot_fallback_reply(prompt: str, lang: str) -> str:
    if lang == "en":
        return (
            "I can handle this locally as Research Copilot. Describe a study goal, say `run the whole demo`, "
            "or ask me to open demo review, real data setup, or Research Agent handoff. Enable an external "
            "provider only when you need open-ended evidence lookup or long-form explanation."
        )
    return (
        "我可以先用本地 Research Copilot 逻辑处理：描述一个研究目标，直接说“跑完整演示”，"
        "或让我打开演示审阅、真实数据配置、Research Agent 交接。只有需要开放式证据检索或长篇解释时，才需要启用外部模型。"
    )


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
    tutorial_requested = any(key in prompt_l for key in ["tutorial", "教程", "step", "步骤", "how do i", "怎么做", "workflow", "流程", "guide", "使用"])
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

    if demo_requested and (viz_requested or "workspace" in prompt_l or "加载" in prompt_l or "populate" in prompt_l):
        add_workflow("workflow_demo_review", "Load Demo Review Workspace", "加载演示审阅工作区", "demo_review")
    elif demo_requested or (extraction_requested and any(key in prompt_l for key in ["start", "开始", "first", "入口"])):
        add_workflow("workflow_demo_extraction", "Start with Demo Extraction", "从演示提取开始", "demo_extraction")

    if extraction_requested and not demo_requested:
        add_workflow("workflow_real_extraction", "Open Real Data Extraction", "打开真实数据提取", "real_extraction")

    if agent_requested:
        add_agent_handoff("agent_handoff", "Hand off to Research Agent", "交给 Research Agent")

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
                elif action.get("kind") == "workflow":
                    _apply_chat_workflow_action(str(action.get("workflow") or ""))
                elif action.get("kind") == "agent_handoff":
                    _prepare_research_agent_handoff_from_ai(st.session_state)
                else:
                    st.session_state["_scroll_to_tab"] = str(action["id"])
                st.rerun()


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
    study = _ensure_copilot_study_state(state)
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])
    concepts = list(config.get("selected_concepts") or [])
    return concepts or ["hr", "map", "temp", "spo2", "sofa2"]


def _apply_copilot_study_to_workspace(state: MutableMapping[str, object]) -> None:
    study = _ensure_copilot_study_state(state)
    question = str(study.get("question") or _copilot_frame_question(study, state.get("language", "en"))).strip()
    if question:
        study["question"] = question
        state["_copilot_last_question"] = question
    state["selected_concepts"] = _copilot_selected_concepts_for_study(state)
    state["_preview_n"] = int(study.get("patient_n") or state.get("demo_mode_patients") or 10)
    state["_copilot_guided_study"] = study


def _apply_chat_workflow_action(workflow: str) -> None:
    """Apply a chat-first command to the live Streamlit workspace."""
    state = st.session_state
    workflow = (workflow or "").strip()
    lang = state.get("language", "en")
    if workflow == "demo_extraction":
        _seed_demo_context_from_chat(state)
        state["_active_main_page"] = "extract"
        state["_assistant_notice"] = (
            "Demo extraction is ready. Confirm the data source, then continue through Cohort and Concepts."
            if lang == "en" else
            "演示提取已准备好。先确认数据源，再继续完成队列和变量步骤。"
        )
    elif workflow == "study_extract":
        _seed_demo_context_from_chat(state)
        _apply_copilot_study_to_workspace(state)
        state["_active_main_page"] = "extract"
        state["_assistant_notice"] = (
            "Research Copilot prepared the classic extraction flow from your guided study."
            if lang == "en" else
            "研究 Copilot 已根据引导式研究准备好经典提取流程。"
        )
    elif workflow == "demo_review":
        _seed_demo_context_from_chat(state)
        state["selected_concepts"] = state.get("selected_concepts") or _copilot_selected_concepts_for_study(state)
        state["_preview_requested"] = True
        mock_params = state.get("mock_params") if isinstance(state.get("mock_params"), dict) else {}
        state["_preview_n"] = int(mock_params.get("n_patients") or state.get("demo_mode_patients") or 10)
        state["_active_main_page"] = "quick_viz"
        state["_assistant_notice"] = (
            "Demo review workspace is loading in Patient Review."
            if lang == "en" else
            "演示审阅工作区正在患者审阅页加载。"
        )
    elif workflow == "study_review":
        _seed_demo_context_from_chat(state)
        _apply_copilot_study_to_workspace(state)
        study = _ensure_copilot_study_state(state)
        state["_preview_requested"] = True
        branch = str(study.get("branch") or "predict")
        target = str(COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"]).get("review_target") or "quick_viz")
        state["_active_main_page"] = target
        state["_assistant_notice"] = (
            "Research Copilot loaded the review workspace from your guided study."
            if lang == "en" else
            "研究 Copilot 已根据引导式研究加载审阅工作区。"
        )
    elif workflow == "real_extraction":
        state["entry_mode"] = "real"
        state["use_mock_data"] = False
        if state.get("database") == "mock":
            state["database"] = "miiv"
        state["_active_main_page"] = "extract"
        state["_assistant_notice"] = (
            "Real Data extraction is open. Choose the database, set the local path, then validate or convert."
            if lang == "en" else
            "真实数据提取已打开。请选择数据库、填写本地路径，然后验证或转换。"
        )
    elif workflow == "study_agent":
        _apply_copilot_study_to_workspace(state)
        _prepare_research_agent_handoff_from_ai(state)
    elif workflow == "guided_demo":
        prompt = "Run the whole demo for me, then stop at the evidence gate."
        reply = _handle_copilot_guided_prompt(prompt, str(lang), state)
        messages = state.setdefault("llm_messages", [])
        if isinstance(messages, list) and reply is not None:
            content, actions = reply
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": content, "actions": actions})
        state["_active_main_page"] = "assistant"
        state["_assistant_notice"] = (
            "Guided Research Copilot demo is ready in the chat."
            if lang == "en" else
            "引导式研究 Copilot 演示已在聊天中准备好。"
        )
    else:
        state["_active_main_page"] = "assistant"
    state["_scroll_to_top"] = True
    state["_inline_ai_panel_open"] = False
    state["_floating_ai_open"] = False
    state.pop("_ai_pending_question", None)


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

def render_llm_settings(
    *,
    expanded: bool = False,
    show_status_card: bool = True,
    controls_only: bool = False,
    show_enable_toggle: bool = True,
    open_sidebar_on_enable: bool = True,
):
    """Render LLM configuration controls in the sidebar or settings popover."""
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

    if show_status_card:
        enabled = bool(st.session_state.llm_enabled)
        status_title = "Research Copilot" if lang == "en" else "研究 Copilot"
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
                <div class="easyicu-ai-sidebar-avatar">AI</div>
                <div>
                    <div class="easyicu-ai-sidebar-title">{status_title}</div>
                    <div class="easyicu-ai-sidebar-subtitle">{status_subtitle}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    label = "AI settings" if lang == "en" else "AI 设置"
    if controls_only:
        _render_llm_settings_controls(
            lang,
            show_enable_toggle=show_enable_toggle,
            open_sidebar_on_enable=open_sidebar_on_enable,
        )
    else:
        with st.expander(label, expanded=expanded):
            _render_llm_settings_controls(
                lang,
                show_enable_toggle=show_enable_toggle,
                open_sidebar_on_enable=open_sidebar_on_enable,
            )


def _render_llm_settings_controls(
    lang: str,
    *,
    show_enable_toggle: bool = True,
    open_sidebar_on_enable: bool = True,
) -> None:
    previous_enabled = bool(st.session_state.llm_enabled)
    if show_enable_toggle:
        enabled = st.toggle(
            "Enable Research Copilot" if lang == "en" else "启用研究 Copilot",
            value=previous_enabled,
            key="_llm_toggle",
        )
        st.session_state.llm_enabled = bool(enabled)
        st.session_state["_floating_ai_open"] = False
        if enabled and open_sidebar_on_enable:
            st.session_state["_sidebar_ai_open"] = True
    else:
        enabled = previous_enabled
        status = (
            "Outbound model calls allowed" if enabled else "Outbound model calls disabled"
        ) if lang == "en" else (
            "已允许模型端调用" if enabled else "模型端调用已关闭"
        )
        if enabled:
            detail = (
                "Shared outbound calls are on; Research Agent still shows a per-run disclosure gate."
                if lang == "en" else
                "共享模型调用已开启；Research Agent 仍会显示单次运行披露关口。"
            )
        else:
            detail = (
                "Provider details can be prepared here; calls still require the shared outbound toggle."
                if lang == "en" else
                "可以先准备服务商配置；是否允许调用由上方共享开关控制。"
            )
        klass = "on" if enabled else "off"
        st.markdown(
            f"""
            <div class="eu-llm-settings-status {klass}">
              <span></span>
              <div><b>{status}</b><p>{detail}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if not enabled and show_enable_toggle:
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
    provider_changed = provider != current_provider
    st.session_state.llm_provider = provider

    p_info = public_provider_defaults(provider)
    if provider_changed:
        _, default_url, default_model, _, _, _ = p_info
        st.session_state.llm_api_key = ""
        st.session_state.llm_base_url = default_url
        st.session_state.llm_model = default_model
        st.session_state["_llm_api_key_inp"] = ""
        st.session_state["_llm_base_url_inp"] = default_url
        st.session_state["_llm_model_inp"] = default_model
        st.session_state.llm_configured = False
        st.rerun()
    desc = p_info[4] if lang == "en" else p_info[5]
    st.caption(desc)

    needs_key = _needs_api_key(provider)

    if needs_key:
        api_key = st.text_input(
            "API Key",
            value=st.session_state.llm_api_key,
            type="password",
            key="_llm_api_key_inp",
            placeholder="sk-...",
        )
        st.session_state.llm_api_key = api_key

    _, default_url, default_model, _, _, _ = p_info
    base_url = st.text_input(
        "API Base URL",
        value=st.session_state.llm_base_url or default_url,
        key="_llm_base_url_inp",
        help="Leave default for standard providers" if lang == "en"
             else "标准服务商保持默认即可",
    )
    st.session_state.llm_base_url = base_url

    model = st.text_input(
        "Model" if lang == "en" else "模型名称",
        value=st.session_state.llm_model or default_model,
        key="_llm_model_inp",
        placeholder=default_model or "model-name",
    )
    st.session_state.llm_model = model

    configured = _is_configured()
    if configured and enabled:
        st.success("Ready" if lang == 'en' else "已就绪")
        st.session_state.llm_configured = True
    elif configured:
        st.info(
            "Provider configured. Turn on outbound model calls above when you want to use it."
            if lang == "en" else
            "服务商配置已就绪；需要使用时请打开上方模型端调用开关。"
        )
        st.session_state.llm_configured = True
    else:
        st.warning("Enter API Key to enable chat"
                   if lang == 'en' else "请输入 API Key")
        st.session_state.llm_configured = False

    st.caption(
        "API keys stay in this browser session only and are never saved by EasyICU."
        if lang == "en" else
        "API Key 只保存在当前浏览器会话中，EasyICU 不会写入本地文件。"
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
            (
                "Research Copilot is disabled. Enable it in the sidebar Copilot settings."
                if lang == "en" else
                "研究 Copilot 当前已关闭。请在侧边栏 Copilot 设置中开启。"
            )
        )
        # Show a brief intro even when disabled
        _render_intro(lang)
        return

    # ---- Guard: not configured -----------------------------------------------
    if not _is_configured():
        st.warning(
            (
                "Please configure your API Key in the sidebar Copilot settings first."
                if lang == "en" else
                "请先在侧边栏 Copilot 设置中设置 API Key。"
            )
        )
        return

    # ---- Guard: openai package missing ---------------------------------------
    try:
        import openai as _openai_mod  # noqa: F401
    except ImportError:
        st.error(
            (
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
        "##### " + ("Chat with Research Copilot" if lang == "en"
                     else "与研究 Copilot 对话")
    )
    st.caption(f"**{provider_name}** · `{model_name}`")
    st.caption(
        "Guided mode: web workflow first, then code and evidence when needed"
        if lang == "en" else
        "引导模式：优先教你如何使用 Web 工作流，再按需补充代码与证据"
    )

    with st.expander(("What can I ask?" if lang == "en" else "我可以问什么？"),
                      expanded=False):
        _render_tips(lang)

    if st.session_state.get("llm_last_tool_events"):
        with st.expander(("Last tool activity" if lang == "en" else "上次工具调用"), expanded=False):
            for event in st.session_state.llm_last_tool_events:
                status = "ok" if event.get("status") == "ok" else "needs attention"
                st.markdown(f"`{event.get('tool', 'tool')}` · {status} — {event.get('detail', '')}")

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

    guided_reply = _handle_copilot_guided_prompt(prompt, lang)
    if guided_reply is not None:
        reply_content, guided_actions = guided_reply
        st.session_state.llm_last_tool_events = []
        st.session_state.llm_last_verification = {
            "status": "pass",
            "issues": [],
        }
        st.session_state.llm_messages.append(
            {
                "role": "assistant",
                "content": reply_content,
                "actions": guided_actions,
            }
        )
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(reply_content)
                _render_nav_actions(guided_actions, key_prefix=f"{key_prefix}_guided")
        return

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

    if st.session_state.get("_active_main_page") == "assistant" and (
        not bool(st.session_state.get("llm_enabled", False)) or not _is_configured()
    ):
        fallback_reply = _local_copilot_fallback_reply(prompt, lang)
        fallback_actions = _suggest_ui_actions(prompt, fallback_reply, lang)
        st.session_state.llm_last_tool_events = []
        st.session_state.llm_last_verification = {
            "status": "pass",
            "issues": [],
        }
        st.session_state.llm_messages.append(
            {"role": "assistant", "content": fallback_reply, "actions": fallback_actions}
        )
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(fallback_reply)
                _render_nav_actions(fallback_actions, key_prefix=f"{key_prefix}_fallback")
        return

    prep_placeholder = st.empty()
    prep_placeholder.info(
        "Preparing tools..." if lang == "en" else "正在准备工具..."
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


def _submit_prompt_background(
    prompt: str,
    lang: str,
    history_container,
    key_prefix: str = "_llm",
) -> None:
    """Append a routed prompt and generate without blocking page navigation."""
    prompt = (prompt or "").strip()
    if not prompt:
        return

    st.session_state.llm_messages.append({"role": "user", "content": prompt})
    with history_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    guided_reply = _handle_copilot_guided_prompt(prompt, lang)
    if guided_reply is not None:
        reply_content, guided_actions = guided_reply
        st.session_state.llm_last_tool_events = []
        st.session_state.llm_last_verification = {
            "status": "pass",
            "issues": [],
        }
        st.session_state.llm_messages.append(
            {
                "role": "assistant",
                "content": reply_content,
                "actions": guided_actions,
            }
        )
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(reply_content)
                _render_nav_actions(guided_actions, key_prefix=f"{key_prefix}_guided")
        return

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

    if st.session_state.get("_active_main_page") == "assistant" and (
        not bool(st.session_state.get("llm_enabled", False)) or not _is_configured()
    ):
        fallback_reply = _local_copilot_fallback_reply(prompt, lang)
        fallback_actions = _suggest_ui_actions(prompt, fallback_reply, lang)
        st.session_state.llm_last_tool_events = []
        st.session_state.llm_last_verification = {
            "status": "pass",
            "issues": [],
        }
        st.session_state.llm_messages.append(
            {"role": "assistant", "content": fallback_reply, "actions": fallback_actions}
        )
        with history_container:
            with st.chat_message("assistant"):
                st.markdown(fallback_reply)
                _render_nav_actions(fallback_actions, key_prefix=f"{key_prefix}_fallback")
        return

    session_id = _start_bg_response(prompt, lang)
    if session_id:
        st.session_state["_ai_bg_session_id"] = session_id
        st.session_state["_ai_bg_responding"] = True
        st.session_state["_ai_bg_response_ready"] = False
        st.session_state["_ai_bg_unread_count"] = 0
        status_text = (
            "Generating response in the background. You can switch pages while I work."
            if lang == "en" else
            "正在后台生成回答。你可以切换页面，助手不会悬浮残留。"
        )
        with history_container:
            st.markdown(
                f'<div class="inline-ai-status-strip">{html.escape(status_text)}</div>',
                unsafe_allow_html=True,
            )
        return

    error_message = (
        "I could not start the assistant response. Check the AI provider settings, then try again."
        if lang == "en" else
        "无法启动助手回答。请检查 AI 服务商设置后重试。"
    )
    st.session_state.llm_messages.append(
        {"role": "assistant", "content": error_message, "actions": []}
    )
    with history_container:
        with st.chat_message("assistant"):
            st.markdown(error_message)


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

    title = "Embedded chat" if lang == "en" else "嵌入式对话"
    with st.expander(title, expanded=expanded):
        st.session_state["_sidebar_ai_open"] = expanded
        if not st.session_state.llm_enabled:
            st.caption(
                "Enable Research Copilot above to start chatting here."
                if lang == "en" else
                "请先在上方开启研究 Copilot，然后在这里对话。"
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

def _inline_ai_context_payload(state, lang: str) -> dict[str, object]:
    is_en = lang == "en"
    mode = str(state.get("entry_mode") or "demo")
    is_demo = mode in {"none", "demo"}
    patient_ids = state.get("patient_ids") or []
    loaded = state.get("loaded_concepts") or {}
    selected = state.get("selected_concepts") or []
    inbound_cohort = state.get("research_agent_inbound_cohort")
    try:
        inbound_rows = len(inbound_cohort) if inbound_cohort is not None else 0
    except TypeError:
        inbound_rows = 0

    if is_demo:
        mock_params = state.get("mock_params") or {}
        patient_count = len(patient_ids) or inbound_rows or int(
            mock_params.get("n_patients") or state.get("demo_mode_patients") or 10
        )
        feature_count = len(loaded) or len(selected)
        context_name = (
            state.get("last_export_name")
            or state.get("research_agent_case_label")
            or "sepsis_mortality_demo"
        )
        mode_label = "demo" if is_en else "演示"
        if feature_count:
            feature_label = f"{feature_count} features" if is_en else f"{feature_count} 个特征"
        else:
            feature_label = "19 modules" if is_en else "19 个模块"
        detail = f"{mode_label} · {patient_count} stays · {feature_label}"
        tags = selected[:6] if selected else ["vitals", "labs", "sofa", "sepsis-3", "lactate", "outcomes"]
    elif patient_ids or inbound_rows or loaded:
        patient_count = len(patient_ids) or inbound_rows
        feature_count = len(loaded) or len(selected)
        context_name = (
            state.get("last_export_name")
            or state.get("research_agent_case_label")
            or ("local cohort loaded" if is_en else "已加载本地队列")
        )
        count_label = f"{patient_count} stays" if patient_count else ("cohort loaded" if is_en else "队列已加载")
        module_label = f"{feature_count} features" if feature_count else ("features pending" if is_en else "特征待确认")
        detail = (
            f"real data · {count_label} · {module_label}"
            if is_en else f"真实数据 · {count_label} · {module_label}"
        )
        tags = selected[:6] or list(loaded)[:6] or ["local export", "evidence-bound"]
    else:
        database = str(state.get("database") or "").strip() or ("local data" if is_en else "本地数据")
        is_mock_source = database.lower() == "mock" or bool(state.get("use_mock_data"))
        context_name = "No cohort loaded" if is_en else "尚未加载队列"
        detail = (
            "mock extraction · waiting for local export"
            if is_mock_source and is_en else
            "模拟提取 · 等待本地导出"
            if is_mock_source else
            "real data · waiting for local export"
            if is_en else
            "真实数据 · 等待本地导出"
        )
        tags = [
            database.upper(),
            "local-only" if is_en else "仅本地",
            "no cohort" if is_en else "无队列",
            "no patient rows" if is_en else "无患者行",
        ]
    return {"context_name": str(context_name), "detail": detail, "tags": [str(tag) for tag in tags[:6]]}


def _inline_ai_context_html(lang: str, state=None) -> str:
    payload = _inline_ai_context_payload(st.session_state if state is None else state, lang)
    is_en = lang == "en"
    tags_html = "".join(f"<span>{html.escape(str(tag))}</span>" for tag in payload["tags"])
    return (
        '<div class="inline-ai-context-card">'
        f'<div class="inline-ai-section-label">{"In context" if is_en else "当前上下文"}</div>'
        f'<h3>{html.escape(str(payload["context_name"]))}</h3>'
        f'<p class="mono">{html.escape(str(payload["detail"]))}</p>'
        f'<div class="inline-ai-tag-row">{tags_html}</div>'
        '</div>'
    )


def _render_inline_ai_blocked_state(lang: str, *, enabled: bool) -> None:
    is_en = lang == "en"
    title = (
        "Assistant is off" if not enabled else "Provider key is missing"
    ) if is_en else (
        "助手已关闭" if not enabled else "缺少模型服务配置"
    )
    desc = (
        "Open Settings and enable the shared AI / API connection before chatting here."
        if not enabled else
        "Configure provider/API key in Settings first. API keys stay in this browser session only."
    ) if is_en else (
        "请先打开设置并启用共享 AI / API 连接，再在这里对话。"
        if not enabled else
        "请先在设置中配置服务商/API Key。API Key 只保存在当前浏览器会话。"
    )
    st.markdown(
        '<div class="inline-ai-blocked">'
        '<div class="inline-ai-blocked-icon">'
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M12 9v4"/><path d="M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>'
        '</svg></div>'
        f'<div><b>{html.escape(title)}</b><p>{html.escape(desc)}</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )


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
    seeded = False
    if not str(state.get("research_agent_question") or "").strip():
        handoff_question = _latest_ai_handoff_question(state)
        if handoff_question:
            state["research_agent_question"] = handoff_question
            state["_research_agent_question_handoff_notice"] = True
            seeded = True

    state["_active_main_page"] = "research_agent"
    state["_ra_view"] = "setup"
    state["_scroll_to_top"] = True
    state["_inline_ai_panel_open"] = False
    state["_floating_ai_open"] = False
    state["_sidebar_ai_open"] = False
    state.pop("_ai_pending_question", None)
    return seeded


def _render_inline_ai_context_and_handoff(lang: str) -> None:
    is_en = lang == "en"
    st.markdown(_inline_ai_context_html(lang), unsafe_allow_html=True)
    st.markdown(
        '<div class="inline-ai-evidence-note">'
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M20 6 9 17l-5-5"/></svg>'
        '<div>'
        f'<b>{html.escape("Evidence-bound" if is_en else "证据绑定")}</b>'
        f'<p>{html.escape("Suggestions only. Drafting stays locked until the agent checks pass." if is_en else "这里只给建议。Agent 检查通过前，草稿始终锁定。")}</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="inline-ai-handoff-card">'
        f'<div class="inline-ai-section-label">{html.escape("Hand off to" if is_en else "交接到")}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2, gap="small")
    with col_a:
        if st.button(
            "Research Agent setup" if is_en else "Research Agent 配置",
            key="_inline_ai_to_agent_setup",
            icon=":material/smart_toy:",
            use_container_width=True,
        ):
            _prepare_research_agent_handoff_from_ai(st.session_state)
            st.rerun()
    with col_b:
        if st.button(
            "How the workflow works" if is_en else "查看工作流说明",
            key="_inline_ai_to_workflow",
            icon=":material/help:",
            use_container_width=True,
        ):
            st.session_state["_active_main_page"] = "tutorial"
            st.session_state["_scroll_to_top"] = True
            st.session_state["_inline_ai_panel_open"] = False
            st.session_state["_floating_ai_open"] = False
            st.session_state.pop("_ai_pending_question", None)
            st.rerun()


def _ai_panel_header_html(lang: str) -> str:
    provider_key = coerce_public_provider(
        st.session_state.get("llm_provider", public_default_provider_key())
    )
    provider_label = public_provider_defaults(provider_key)[0] or provider_key
    default_model = public_provider_defaults(provider_key)[2] or "model"
    model_label = st.session_state.get("llm_model") or default_model
    title = "Research Copilot" if lang == "en" else "研究 Copilot"
    if provider_key == "easyicu_hosted":
        subtitle = (
            "EasyICU hosted · evidence-bound"
            if lang == "en" else
            "EasyICU 托管 · 证据绑定"
        )
    else:
        subtitle = (
            f"{provider_label} · {model_label} · evidence-bound"
            if lang == "en" else
            f"{provider_label} · {model_label} · 证据绑定"
        )
    llm_configured = _is_configured()
    if not st.session_state.get("llm_enabled", False):
        status_label = "AI off" if lang == "en" else "AI 已关闭"
        status_color = "var(--ink-4)"
    elif not llm_configured:
        status_label = "key missing" if lang == "en" else "缺少 API key"
        status_color = "var(--warn)"
    else:
        status_label = "Ready" if lang == "en" else "已就绪"
        status_color = "var(--ok)"
    status_pill_html = (
        f'<span class="inline-ai-status-pill eu-pill mono" '
        f'style="font-size:10px;padding:2px 7px;height:18px;margin-left:8px;'
        f'background:var(--surface);color:{status_color};'
        f'border:1px solid var(--hair-2);vertical-align:middle">'
        f'<span class="dot" style="background:{status_color}"></span>'
        f'{html.escape(status_label)}</span>'
    )
    return (
        '<div class="inline-ai-header">'
        '<div class="inline-ai-avatar">AI</div>'
        '<div class="inline-ai-meta">'
        f'<div class="inline-ai-title">{html.escape(title)}</div>'
        f'<div class="inline-ai-subtitle">{html.escape(subtitle)}</div>'
        '</div>'
        f'{status_pill_html}'
        '</div>'
    )


def render_inline_ai_panel(*, force_open: bool = False, allow_close: bool = True):
    """Render the non-floating AI assistant as part of the main workspace."""
    _init_chat_state()
    lang = st.session_state.get("language", "en")
    pending_prompt = bool(st.session_state.get("_ai_pending_question"))
    active_page = st.session_state.get("_active_main_page")
    if active_page != "assistant":
        st.session_state["_inline_ai_panel_open"] = False
        st.session_state["_floating_ai_open"] = False
        if force_open or pending_prompt:
            st.session_state["_active_main_page"] = "assistant"
            st.session_state["_scroll_to_top"] = True
            st.rerun()
        return
    if force_open:
        st.session_state["_inline_ai_panel_open"] = True
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
            position: relative;
            border: 1px solid var(--hair);
            border-radius: var(--r-3);
            background: color-mix(in oklab, white 74%, var(--surface));
            padding: 0;
            margin: 0.72rem 0 1.35rem;
            overflow: hidden;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-header {
            display: flex;
            align-items: center;
            gap: 0.78rem;
            min-width: 0;
            padding: 1.02rem 4.1rem 1rem 1.18rem;
            border-bottom: 1px solid var(--hair);
        }
        .stApp div.st-key-inline_ai_assistant_panel .st-key-_inline_ai_close {
            position: absolute !important;
            top: 0.66rem;
            right: 0.86rem;
            z-index: 5;
            width: 2.55rem !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel .st-key-_inline_ai_close button {
            width: 2.55rem !important;
            min-width: 2.55rem !important;
            min-height: 2.05rem !important;
            padding: 0 !important;
            border-radius: var(--r-2) !important;
            background: white !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-avatar {
            width: 2.25rem;
            height: 2.25rem;
            border-radius: var(--r-2);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--accent);
            background: var(--accent-soft);
            border: 1px solid color-mix(in oklab, var(--accent) 30%, var(--hair));
            font-family: var(--font-mono);
            font-size: 0.76rem;
            font-weight: 760;
            flex: 0 0 auto;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-title {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 720;
            line-height: 1.2;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-subtitle {
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 0.75rem;
            line-height: 1.35;
            margin-top: 0.1rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-welcome {
            border: 0;
            border-radius: 0;
            background: transparent;
            padding: 0.2rem 0 0.25rem;
            margin: 0;
            box-shadow: none;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-welcome-title {
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 0.75rem;
            font-weight: 650;
            letter-spacing: .12em;
            text-transform: uppercase;
            line-height: 1.25;
            margin: 0 0 0.5rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-welcome-subtitle,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-user-bubble,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-answer-card {
            color: var(--ink);
            font-size: 0.9rem;
            line-height: 1.58;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-sample {
            display: grid;
            grid-template-columns: minmax(0, 1fr);
            gap: 0.72rem;
            margin: 0.7rem 0 0.78rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-user-bubble,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-answer-card,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation {
            border-radius: var(--r-3);
            border: 1px solid var(--hair);
            background: var(--surface);
            padding: 0.86rem 1rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-user-bubble {
            max-width: 68%;
            margin-left: auto;
            background: var(--ink);
            color: white;
            border-color: var(--ink);
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-answer-card {
            max-width: 82%;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation {
            display: grid;
            gap: 0.32rem;
            max-width: 82%;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation span {
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation strong,
        .stApp div.st-key-inline_ai_assistant_panel .floating-ai-welcome-hint {
            color: var(--ink);
            font-size: 0.86rem;
            line-height: 1.5;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-blocked,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-context-card,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-evidence-note,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-handoff-card {
            margin: 0.78rem 1.18rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-blocked {
            display: grid;
            grid-template-columns: 2.25rem 1fr;
            gap: 0.82rem;
            align-items: start;
            padding: 0.9rem 1rem;
            border: 1px solid color-mix(in oklab, var(--warn) 32%, var(--hair));
            border-radius: var(--r-3);
            background: color-mix(in oklab, var(--warn) 10%, white);
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-blocked-icon {
            width: 2.1rem;
            height: 2.1rem;
            border-radius: var(--r-2);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--warn);
            background: white;
            border: 1px solid color-mix(in oklab, var(--warn) 28%, var(--hair));
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-blocked b,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-context-card h3,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-evidence-note b {
            color: var(--ink);
            font-size: 0.95rem;
            line-height: 1.3;
            font-weight: 720;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-blocked p,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-context-card p,
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-evidence-note p {
            margin: 0.24rem 0 0;
            color: var(--ink-2);
            font-size: 0.84rem;
            line-height: 1.5;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-context-card {
            padding: 1rem;
            border: 1px solid var(--hair);
            border-radius: var(--r-3);
            background: white;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-section-label {
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 0.72rem;
            font-weight: 650;
            letter-spacing: .12em;
            line-height: 1.2;
            text-transform: uppercase;
            margin-bottom: 0.7rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-context-card h3 {
            margin: 0;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-tag-row {
            display: flex;
            gap: 0.42rem;
            flex-wrap: wrap;
            margin-top: 0.82rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-tag-row span {
            display: inline-flex;
            align-items: center;
            min-height: 24px;
            padding: 0 0.72rem;
            border: 1px solid var(--hair);
            border-radius: 999px;
            background: var(--surface);
            color: var(--ink-2);
            font-family: var(--font-mono);
            font-size: 0.78rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-evidence-note {
            display: grid;
            grid-template-columns: 1.3rem 1fr;
            gap: 0.72rem;
            align-items: start;
            padding: 0.98rem 1rem;
            border: 1px solid color-mix(in oklab, var(--ok) 30%, var(--hair));
            border-radius: var(--r-3);
            background: color-mix(in oklab, var(--ok) 13%, white);
            color: var(--ok);
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-handoff-card {
            margin-bottom: 0.2rem;
            padding-top: 0.2rem;
        }
        .stApp div.st-key-inline_ai_assistant_panel .inline-ai-status-strip {
            margin: 0.42rem 1.18rem 0.66rem;
            padding: 0.62rem 0.78rem;
            border: 1px solid var(--hair);
            border-radius: var(--r-2);
            background: color-mix(in oklab, var(--surface) 86%, white);
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 0.72rem;
            line-height: 1.35;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"] {
            display: flex !important;
            align-items: flex-start !important;
            gap: 0.72rem !important;
            padding: 0.36rem 1.18rem !important;
            background: transparent !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse !important;
            justify-content: flex-start !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarUser"],
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarAssistant"] {
            width: 2rem !important;
            height: 2rem !important;
            min-width: 2rem !important;
            border-radius: var(--r-2) !important;
            border: 1px solid var(--hair) !important;
            box-shadow: none !important;
            overflow: hidden !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarAssistant"] {
            background: var(--accent-soft) !important;
            color: var(--accent) !important;
            border-color: color-mix(in oklab, var(--accent) 26%, var(--hair)) !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarUser"] {
            background: color-mix(in oklab, var(--warn) 13%, white) !important;
            color: var(--ink) !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarUser"] > *,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarAssistant"] > * {
            display: none !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarUser"]::after,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarAssistant"]::after {
            display: grid !important;
            place-items: center !important;
            width: 100% !important;
            height: 100% !important;
            font-family: var(--font-mono) !important;
            font-size: 0.56rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.04em !important;
            line-height: 1 !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarUser"]::after {
            content: "YOU";
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageAvatarAssistant"]::after {
            content: "AI";
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] {
            flex: 0 1 auto !important;
            width: fit-content !important;
            max-width: min(820px, 82%) !important;
            min-width: 0 !important;
            margin: 0 !important;
            padding: 0.84rem 0.98rem !important;
            border: 1px solid var(--hair) !important;
            border-radius: var(--r-3) !important;
            background: var(--surface) !important;
            color: var(--ink) !important;
            box-shadow: none !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"]::before {
            content: "Assistant";
            display: block;
            margin-bottom: 0.34rem;
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 0.68rem;
            font-weight: 680;
            letter-spacing: 0.12em;
            line-height: 1.1;
            text-transform: uppercase;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
            max-width: min(650px, 68%) !important;
            background: var(--ink) !important;
            border-color: var(--ink) !important;
            color: white !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"]::before {
            content: "You";
            color: rgba(255, 255, 255, 0.68);
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] li {
            color: white !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] p,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] li,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stMarkdownContainer"] p,
        .stApp div.st-key-inline_ai_assistant_panel input,
        .stApp div.st-key-inline_ai_assistant_panel label {
            font-size: 0.9rem !important;
            line-height: 1.56;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] p,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] ul,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] ol {
            margin-top: 0 !important;
            margin-bottom: 0.48rem !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] p:last-child,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] ul:last-child,
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"] ol:last-child {
            margin-bottom: 0 !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel .stButton > button,
        .stApp div.st-key-inline_ai_assistant_panel form button {
            min-height: 38px;
            border-radius: var(--r-2);
            font-size: 0.83rem;
            font-weight: 650;
            box-shadow: none !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] {
            border: 0 !important;
            padding: 0 1.18rem 1.05rem !important;
            background: transparent !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: flex-end !important;
            gap: 0.5rem !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] [data-testid="stColumn"] {
            min-width: 0 !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] [data-testid="stColumn"]:last-child {
            flex: 0 0 3.25rem !important;
            width: 3.25rem !important;
            min-width: 3.25rem !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] [data-testid="stTextInput"] {
            margin-bottom: 0 !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] input {
            min-height: 40px !important;
            border-radius: var(--r-2) !important;
        }
        .stApp div.st-key-inline_ai_assistant_panel [data-testid="stFormSubmitButton"] button {
            min-width: 42px !important;
            min-height: 40px !important;
            padding-left: 0.7rem !important;
            padding-right: 0.7rem !important;
            font-size: 1rem !important;
        }
        @media (max-width: 900px) {
            .stApp div.st-key-inline_ai_assistant_panel { margin-top: 0.55rem; }
            .stApp div.st-key-inline_ai_assistant_panel .inline-ai-header { padding: 0.86rem 3.85rem 0.86rem 0.86rem; }
            .stApp div.st-key-inline_ai_assistant_panel .st-key-_inline_ai_close {
                top: -2.65rem;
                right: 0.78rem;
            }
            .stApp div.st-key-inline_ai_assistant_panel .floating-ai-user-bubble,
            .stApp div.st-key-inline_ai_assistant_panel .floating-ai-answer-card,
            .stApp div.st-key-inline_ai_assistant_panel .floating-ai-recommendation { max-width: 100%; }
            .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"] {
                padding-left: 0.86rem !important;
                padding-right: 0.86rem !important;
            }
            .stApp div.st-key-inline_ai_assistant_panel .inline-ai-status-strip {
                margin-left: 0.86rem;
                margin-right: 0.86rem;
            }
            .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessageContent"],
            .stApp div.st-key-inline_ai_assistant_panel [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
                max-width: calc(100% - 2.72rem) !important;
            }
            .stApp div.st-key-inline_ai_assistant_panel .inline-ai-blocked,
            .stApp div.st-key-inline_ai_assistant_panel .inline-ai-context-card,
            .stApp div.st-key-inline_ai_assistant_panel .inline-ai-evidence-note,
            .stApp div.st-key-inline_ai_assistant_panel .inline-ai-handoff-card {
                margin-left: 0.86rem;
                margin-right: 0.86rem;
            }
            .stApp div.st-key-inline_ai_assistant_panel [data-testid="stForm"] {
                padding-left: 0.86rem !important;
                padding-right: 0.86rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="inline_ai_assistant_panel"):
        if allow_close:
            left, close_col = st.columns([10, 1], gap="small")
        else:
            left = st.container()
            close_col = None
        with left:
            st.markdown(_ai_panel_header_html(lang), unsafe_allow_html=True)
        if close_col is not None:
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

        enabled = bool(st.session_state.get("llm_enabled", False))
        configured = _is_configured()
        if not enabled:
            _render_inline_ai_blocked_state(lang, enabled=False)
            _render_inline_ai_context_and_handoff(lang)
            return

        if not configured:
            _render_inline_ai_blocked_state(lang, enabled=True)
            _render_inline_ai_context_and_handoff(lang)
            return

        _render_compact_chat_panel(
            lang=lang,
            panel_key="_llm_inline_workspace",
            history_height=440,
            show_starters=not pending_prompt,
        )
        _render_inline_ai_context_and_handoff(lang)


def _render_ai_assistant_workspace_page(lang: str, *, pending_prompt: bool) -> None:
    with st.container(key="ai_assistant_page_panel"):
        st.markdown('<div class="eu-copilot-page-marker"></div>', unsafe_allow_html=True)
        st.markdown(_ai_panel_header_html(lang), unsafe_allow_html=True)

        rail_col, chat_col, state_col = st.columns([0.7, 1.58, 0.92], gap="large")
        with rail_col:
            _render_copilot_session_rail(lang)

        with chat_col:
            _render_copilot_quick_actions(lang)
            st.markdown(
                '<div class="inline-ai-section-label">'
                f'{html.escape("Conversation" if lang == "en" else "对话")}'
                '</div>',
                unsafe_allow_html=True,
            )
            if not bool(st.session_state.get("llm_enabled", False)) or not _is_configured():
                note = (
                    "Local guided mode is available now. Enable an external provider only for free-form evidence lookup."
                    if lang == "en" else
                    "本地引导模式现在即可使用。只有需要开放式证据检索时才需要启用外部模型。"
                )
                st.markdown(
                    f'<div class="inline-ai-status-strip">{html.escape(note)}</div>',
                    unsafe_allow_html=True,
                )
            _render_compact_chat_panel(
                lang=lang,
                panel_key="_llm_ai_page_workspace",
                history_height=620,
                show_starters=not pending_prompt,
                background_pending_prompts=True,
            )

        with state_col:
            _render_copilot_stage_workspace(lang)
            _render_copilot_state_panel(lang)
            _render_inline_ai_context_and_handoff(lang)


def _render_copilot_quick_actions(lang: str) -> None:
    """Render chat-first commands that also drive the classic workspace."""
    is_en = lang == "en"
    st.markdown(
        '<div class="inline-ai-section-label">'
        f'{html.escape("Start here" if is_en else "从这里开始")}'
        '</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(5, gap="small")
    actions = [
        (
            "Guided demo" if is_en else "引导演示",
            "guided_demo",
            ":material/auto_awesome:",
            "_copilot_guided_demo",
            "Run the chat-first demo to the evidence gate" if is_en else "用聊天方式跑到证据闸门",
        ),
        (
            "Demo extraction" if is_en else "演示提取",
            "demo_extraction",
            ":material/science:",
            "_copilot_demo_extraction",
            "Start the classic 4-step flow with demo data" if is_en else "用演示数据进入经典四步流程",
        ),
        (
            "Demo review" if is_en else "演示审阅",
            "demo_review",
            ":material/table_chart:",
            "_copilot_demo_review",
            "Load a lightweight Patient Review workspace" if is_en else "加载轻量患者审阅工作区",
        ),
        (
            "Real data" if is_en else "真实数据",
            "real_extraction",
            ":material/database:",
            "_copilot_real_extraction",
            "Open local data-source setup" if is_en else "打开本地真实数据源配置",
        ),
        (
            "Agent setup" if is_en else "Agent 配置",
            "agent_handoff",
            ":material/smart_toy:",
            "_copilot_agent_setup",
            "Hand the latest question to Research Agent" if is_en else "把最近的问题交给 Research Agent",
        ),
    ]
    for col, (label, workflow, icon_name, key, help_text) in zip(cols, actions):
        with col:
            if st.button(
                label,
                key=key,
                icon=icon_name,
                use_container_width=True,
                help=help_text,
            ):
                if workflow == "agent_handoff":
                    _prepare_research_agent_handoff_from_ai(st.session_state)
                else:
                    _apply_chat_workflow_action(workflow)
                st.rerun()


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


def _copilot_stage_detail(study: MutableMapping[str, object], step: str, lang: str) -> str:
    is_en = lang == "en"
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])
    if step == "question":
        return str(study.get("question") or (config["chip"] if is_en else config["question_zh"]))
    if step == "data":
        mode = str(study.get("data_mode") or "demo")
        if mode == "real":
            return "local data · no upload" if is_en else "本地数据 · 不上传"
        return "demo data · local only" if is_en else "演示数据 · 仅本机"
    if step == "cohort":
        if branch == "crossdb":
            return f"{int(study.get('db_count') or 6)} databases · shared sepsis definition" if is_en else f"{int(study.get('db_count') or 6)} 个数据库 · 共享脓毒症定义"
        return f"{int(study.get('patient_n') or 10)} stays · first ICU stay · first 24h" if is_en else f"{int(study.get('patient_n') or 10)} 例 stay · 首次 ICU · 前 24h"
    if step == "concepts":
        modules = list(study.get("modules") or COPILOT_DEFAULT_MODULES)
        return f"{len(modules)} modules · coverage audited before analysis" if is_en else f"{len(modules)} 个模块 · 分析前审计覆盖率"
    if step == "extract":
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
        return "locked until checks pass and a human signs off" if is_en else "证据检查和人工确认前保持锁定"
    return ""


def _copilot_stage_card_html(
    study: MutableMapping[str, object],
    step: str,
    lang: str,
    *,
    compact: bool = False,
) -> str:
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
    if compact and status == "done":
        return (
            f'<div class="eu-copilot-stage collapsed {status}">'
            '<span class="mark"></span>'
            f'<b>{html.escape(_copilot_step_label(step, lang))}</b>'
            f'<em>{html.escape(_copilot_stage_detail(study, step, lang))}</em>'
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


def _render_copilot_session_rail(lang: str) -> None:
    """Render the Claude/Codex-style session rail for the standalone Copilot page."""
    is_en = lang == "en"
    state = st.session_state
    study = _ensure_copilot_study_state(state)
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])
    question = str(study.get("question") or _copilot_frame_question(study, lang)).strip()
    branch_label = str(config["chip"] if is_en else config["question_zh"])
    data_mode = str(study.get("data_mode") or "demo")
    rail_html = (
        '<div class="eu-copilot-rail">'
        f'<div class="inline-ai-section-label">{html.escape("Sessions" if is_en else "研究会话")}</div>'
        '<div class="eu-copilot-session live">'
        '<span class="dot"></span><div>'
        f'<b>{html.escape(branch_label)}</b>'
        f'<p>{html.escape(question[:120] or branch_label)}</p>'
        '</div></div>'
        '<div class="eu-copilot-session">'
        '<span class="dot muted"></span><div>'
        f'<b>{html.escape("Evidence gate" if is_en else "证据闸门")}</b>'
        f'<p>{html.escape("Draft locked until checks pass" if is_en else "证据检查通过前草稿锁定")}</p>'
        '</div></div>'
        '<div class="eu-copilot-rail-meta">'
        f'<span>{html.escape("mode" if is_en else "模式")}<b>{html.escape("demo" if data_mode == "demo" else "real")}</b></span>'
        f'<span>{html.escape("patients" if is_en else "患者")}<b>{int(study.get("patient_n") or 10)}</b></span>'
        f'<span>{html.escape("modules" if is_en else "模块")}<b>{len(study.get("modules") or COPILOT_DEFAULT_MODULES)}</b></span>'
        '</div>'
        '</div>'
    )
    st.markdown(rail_html, unsafe_allow_html=True)
    if st.button(
        "New study" if is_en else "新研究",
        key="_copilot_new_study",
        icon=":material/add:",
        use_container_width=True,
    ):
        _reset_copilot_study_state(state)
        state.pop("_ai_pending_question", None)
        st.rerun()
    if st.button(
        "Run whole demo" if is_en else "跑完整演示",
        key="_copilot_rail_guided_demo",
        icon=":material/auto_awesome:",
        use_container_width=True,
    ):
        _apply_chat_workflow_action("guided_demo")
        st.rerun()


def _render_copilot_stage_workspace(lang: str) -> None:
    """Render guided-study cards adapted from the Claude Design prototype."""
    is_en = lang == "en"
    state = st.session_state
    study = _ensure_copilot_study_state(state)
    branch = str(study.get("branch") or "predict")
    config = COPILOT_BRANCH_CONFIG.get(branch, COPILOT_BRANCH_CONFIG["predict"])
    active_step = str(study.get("step") or "question")
    active_idx = COPILOT_STEP_INDEX.get(active_step, 0)
    stages = []
    for step, _label in COPILOT_STUDY_STEPS:
        status = _copilot_stage_status(study, step)
        stages.append(_copilot_stage_card_html(study, step, lang, compact=status == "done" and step != active_step))
    checks = [
        ("Denominators resolved", "分母已解析", active_idx >= COPILOT_STEP_INDEX["review"]),
        ("Coverage audited", "覆盖率已审计", active_idx >= COPILOT_STEP_INDEX["analysis"]),
        ("Artifacts traceable", "产物可追溯", active_idx >= COPILOT_STEP_INDEX["draft"]),
        ("Reviewer sign-off", "人工确认", bool(study.get("draft_signed"))),
    ]
    gate_rows = "".join(
        '<div class="eu-copilot-gate-row {status}"><span></span><b>{label}</b><em>{state}</em></div>'.format(
            status="done" if done else "pending",
            label=html.escape(label_en if is_en else label_zh),
            state=html.escape(("passed" if done else "pending") if is_en else ("通过" if done else "待确认")),
        )
        for label_en, label_zh, done in checks
    )
    st.markdown(
        '<div class="eu-copilot-study-workspace">'
        '<div class="eu-copilot-study-head">'
        f'<div class="inline-ai-section-label">{html.escape("Study plan" if is_en else "研究计划")}</div>'
        f'<h3>{html.escape(config["chip"] if is_en else config["question_zh"])}</h3>'
        f'<p>{html.escape(str(study.get("question") or _copilot_frame_question(study, lang)))}</p>'
        '</div>'
        '<div class="eu-copilot-stage-list">'
        + "".join(stages)
        + '</div>'
        '<div class="eu-copilot-gate">'
        f'<div class="inline-ai-section-label">{html.escape("Evidence gate" if is_en else "证据闸门")}</div>'
        + gate_rows
        + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    action_cols = st.columns(2, gap="small")
    with action_cols[0]:
        if st.button(
            "Open review" if is_en else "打开审阅",
            key="_copilot_stage_open_review",
            icon=":material/preview:",
            use_container_width=True,
        ):
            _apply_chat_workflow_action("study_review")
            st.rerun()
    with action_cols[1]:
        if st.button(
            "Agent setup" if is_en else "Agent 配置",
            key="_copilot_stage_agent_setup",
            icon=":material/assignment:",
            use_container_width=True,
        ):
            _apply_chat_workflow_action("study_agent")
            st.rerun()


def _render_copilot_state_panel(lang: str) -> None:
    """Show the shared GUI/chat state so users can see both architectures linked."""
    is_en = lang == "en"
    state = st.session_state
    entry_mode = str(state.get("entry_mode") or "none")
    database = str(state.get("database") or "miiv")
    selected_count = len(state.get("selected_concepts") or [])
    loaded_count = len(state.get("loaded_concepts") or {})
    patient_count = len(state.get("patient_ids") or [])
    data_path = str(state.get("data_path") or "").strip()
    path_validated = bool(state.get("path_validated"))
    study = _ensure_copilot_study_state(state)
    study_question = str(study.get("question") or "").strip()
    study_step = str(study.get("step") or "question")
    study_idx = COPILOT_STEP_INDEX.get(study_step, 0)
    if entry_mode == "none" and study.get("data_mode") in {"demo", "real"}:
        entry_mode = str(study.get("data_mode"))

    mode_label = {
        "demo": "Demo" if is_en else "演示",
        "real": "Real Data" if is_en else "真实数据",
        "none": "Not selected" if is_en else "未选择",
    }.get(entry_mode, entry_mode)
    db_label = "mock" if database == "mock" else database.upper()

    step1_done = entry_mode == "demo" or bool(data_path and path_validated) or study_idx >= COPILOT_STEP_INDEX["data"]
    step2_done = bool(state.get("step2_confirmed")) or study_idx >= COPILOT_STEP_INDEX["cohort"]
    step3_done = bool(state.get("step3_confirmed")) or selected_count > 0 or study_idx >= COPILOT_STEP_INDEX["concepts"]
    review_done = loaded_count > 0 or patient_count > 0 or study_idx >= COPILOT_STEP_INDEX["review"]
    agent_ready = bool(state.get("research_agent_question")) or review_done or study_idx >= COPILOT_STEP_INDEX["analysis"]

    def status(done: bool, active: bool = False) -> str:
        if done:
            return "done"
        return "active" if active else "todo"

    rows = [
        _workflow_status_step(
            "Question" if is_en else "研究问题",
            (str(state.get("research_agent_question") or study_question).strip()[:82] or (
                "Ask in chat or choose a starter prompt" if is_en else "可在聊天中提出，也可选择起始问题"
            )),
            status(bool(state.get("research_agent_question") or study_question), active=study_step == "question"),
        ),
        _workflow_status_step(
            "Data source" if is_en else "数据源",
            f"{mode_label} · {db_label}" + ("" if not data_path else f" · {Path(data_path).name}"),
            status(step1_done, active=study_step == "data" or entry_mode == "none"),
        ),
        _workflow_status_step(
            "Cohort" if is_en else "队列",
            ("confirmed" if step2_done else "waiting for Step 2") if is_en else ("已确认" if step2_done else "等待第 2 步"),
            status(step2_done, active=study_step == "cohort" or (step1_done and not step2_done)),
        ),
        _workflow_status_step(
            "Concepts" if is_en else "变量",
            (f"{selected_count or len(study.get('modules') or [])} selected" if is_en else f"已选 {selected_count or len(study.get('modules') or [])} 个"),
            status(step3_done, active=study_step == "concepts" or (step2_done and not step3_done)),
        ),
        _workflow_status_step(
            "Review" if is_en else "审阅",
            f"{loaded_count} concepts · {patient_count} patients" if is_en else f"{loaded_count} 个概念 · {patient_count} 位患者",
            status(review_done, active=study_step == "review" or (step3_done and not review_done)),
        ),
        _workflow_status_step(
            "Agent" if is_en else "Agent",
            ("draft gated" if study_step == "draft" else "ready for handoff") if is_en else ("草稿已闸门锁定" if study_step == "draft" else "可交接"),
            status(agent_ready, active=study_step in {"analysis", "draft"} or review_done),
        ),
    ]

    st.markdown(
        '<div class="eu-copilot-state-card">'
        f'<div class="inline-ai-section-label">{html.escape("Workspace state" if is_en else "工作区状态")}</div>'
        f'<h3>{html.escape("Chat and GUI stay linked" if is_en else "聊天和图形界面保持联动")}</h3>'
        f'<p>{html.escape("Every chat command writes into the same session state as the classic EasyICU panels." if is_en else "每个聊天命令都会写入经典 EasyICU 面板共用的会话状态。")}</p>'
        '<div class="eu-copilot-stepper">'
        + "".join(rows)
        + '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    nav_cols = st.columns(2, gap="small")
    with nav_cols[0]:
        if st.button(
            "Classic flow" if is_en else "经典流程",
            key="_copilot_open_classic_flow",
            icon=":material/account_tree:",
            use_container_width=True,
        ):
            state["_active_main_page"] = "extract"
            state["_scroll_to_top"] = True
            st.rerun()
    with nav_cols[1]:
        if st.button(
            "Patient Review" if is_en else "患者审阅",
            key="_copilot_open_patient_review",
            icon=":material/table_chart:",
            use_container_width=True,
        ):
            state["_active_main_page"] = "quick_viz"
            state["_scroll_to_top"] = True
            st.rerun()


def render_ai_assistant_page(lang: str | None = None) -> None:
    """Render the latest Tools · Research Copilot page using the live chat backend."""
    _init_chat_state()
    lang = lang or st.session_state.get("language", "en")
    is_en = lang == "en"
    pending_prompt = bool(st.session_state.get("_ai_pending_question"))
    provider_key = coerce_public_provider(
        st.session_state.get("llm_provider", public_default_provider_key())
    )
    if not _needs_api_key(provider_key):
        st.session_state["llm_enabled"] = True
        st.session_state["_llm_toggle"] = True
    st.session_state["_floating_ai_open"] = False
    st.session_state["_inline_ai_panel_open"] = False
    st.markdown(
        '<div class="eu-ai-page-head">'
        f'<div class="eyebrow">{html.escape("EasyICU · chat-first research workspace" if is_en else "EasyICU · 聊天优先研究工作台")}</div>'
        f'<h1>{html.escape("Research Copilot" if is_en else "研究 Copilot")}</h1>'
        f'<p>{html.escape("A parallel chat-first interface for framing a study, driving the classic workspace, and handing a vetted question into Research Agent." if is_en else "一套并行的聊天优先入口：可整理研究问题、驱动经典界面，并把审核后的问题交给 Research Agent。")}</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    _render_ai_assistant_workspace_page(lang, pending_prompt=pending_prompt)


def _starter_prompts(lang: str) -> list[str]:
    if lang == "en":
        return [
            "Predict sepsis mortality using first 24h lactate, SOFA, vitals, and labs.",
            "Compare a sepsis mortality signal across ICU databases.",
            "Audit data quality first: missingness, coverage, out-of-range values, and sparse modules.",
            "Run the whole demo for me, then stop at the evidence gate.",
        ]
    return [
        "用前 24 小时乳酸、SOFA、生命体征和实验室指标预测脓毒症死亡。",
        "跨 ICU 数据库比较脓毒症死亡信号是否复现。",
        "先做数据质量审计：缺失率、覆盖率、越界值和稀疏模块。",
        "帮我跑完整演示，然后停在证据闸门。",
    ]


def _render_chat_welcome(*, lang: str, panel_key: str, history_container, show_starters: bool = True) -> None:
    title = "Research Copilot" if lang == "en" else "研究 Copilot"
    subtitle = (
        "I can walk a whole study through by chat: frame the question, pick data, assemble the cohort, load review, and hand a gated question to Research Agent."
        if lang == "en" else
        "我可以用聊天带你走完整研究：框定问题、选择数据、组装队列、加载审阅，并把带闸门的问题交给 Research Agent。"
    )
    prompt_hint = (
        "Try one of these prompts:" if lang == "en" else "可以从这些提示开始："
    ) if show_starters else (
        "Ask about your current workflow, cohort, or export settings."
        if lang == "en" else
        "可以直接询问当前流程、队列筛选或导出设置。"
    )
    sample_q = (
        'Help me turn "does lactate matter for sepsis mortality" into a researchable question.'
        if lang == "en" else
        "帮我把“乳酸是否影响脓毒症死亡率”整理成可研究问题。"
    )
    sample_a = (
        "I will bind it to a cohort, outcome, first-24h window, feature modules, review gate, and Research Agent handoff."
        if lang == "en" else
        "我会把它绑定到队列、结局、前 24 小时窗口、特征模块、审阅闸门和 Research Agent 交接。"
    )
    rec_label = "Evidence-bound note" if lang == "en" else "证据绑定提示"
    rec_text = (
        "I will not assert effect sizes here; those come from the run, and only after the evidence gate opens."
        if lang == "en" else
        "我不会在这里断言效应量；这些必须来自实际运行，并且只有证据闸门通过后才能写入。"
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


def _render_compact_chat_panel(
    *,
    lang: str,
    panel_key: str,
    history_height: int = 320,
    show_starters: bool = True,
    background_pending_prompts: bool = False,
) -> None:
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
                role = str(msg.get("role") or "assistant")
                avatar = ":material/person:" if role == "user" else ":material/smart_toy:"
                with st.chat_message(role, avatar=avatar):
                    st.markdown(msg["content"])
                    if role == "assistant" and msg.get("actions"):
                        _render_nav_actions(msg["actions"], key_prefix=f"{panel_key}_{msg_idx}")
            if queued_prompt:
                status_text = (
                    "Using page context to ask the assistant..."
                    if lang == "en" else
                    "正在带着当前页面上下文向 AI 提问..."
                )
                st.markdown(
                    f'<div class="inline-ai-status-strip">{html.escape(status_text)}</div>',
                    unsafe_allow_html=True,
                )
                if background_pending_prompts:
                    _submit_prompt_background(
                        queued_prompt,
                        lang,
                        history_container,
                        key_prefix=panel_key,
                    )
                else:
                    _submit_prompt(queued_prompt, lang, history_container, key_prefix=panel_key)
            elif st.session_state.get("_ai_bg_responding", False):
                status_text = (
                    "Generating response in the background. You can continue elsewhere in EasyICU."
                    if lang == "en" else
                    "正在后台生成回答。你可以继续使用 EasyICU 的其他页面。"
                )
                st.markdown(
                    f'<div class="inline-ai-status-strip">{html.escape(status_text)}</div>',
                    unsafe_allow_html=True,
                )

    with st.form(f"{panel_key}_form", clear_on_submit=True):
        input_col, send_col = st.columns([1, 0.08], gap="small")
        with input_col:
            prompt = st.text_input(
                "Ask EasyICU AI" if lang == "en" else "向 EasyICU AI 提问",
                placeholder="Ask about the current workflow..." if lang == "en" else "询问当前流程、概念或报错...",
                label_visibility="collapsed",
            )
        with send_col:
            send_clicked = st.form_submit_button(
                "→",
                type="primary",
                use_container_width=True,
                help="Send" if lang == "en" else "发送",
            )
    if send_clicked and prompt.strip():
        st.session_state["_ai_pending_question"] = prompt.strip()
        st.rerun()

    if st.session_state.llm_messages:
        action_cols = st.columns(2)
        with action_cols[0]:
            st.download_button(
                "Export Chat" if lang == "en" else "导出对话",
                data=_build_chat_export_text(),
                file_name=f"easyicu_ai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
                key=f"{panel_key}_export_chat",
                icon=":material/download:",
            )
        with action_cols[1]:
            if st.button(
                "Clear Chat" if lang == "en" else "清空对话",
                key=f"{panel_key}_clear_chat",
                use_container_width=True,
                icon=":material/delete:",
            ):
                st.session_state.llm_messages = []
                st.session_state["_ai_pending_question"] = None
                st.rerun()


def _render_floating_copilot_context_actions(lang: str) -> None:
    """Render compact context and workspace-driving actions for the global dock."""
    is_en = lang == "en"
    st.markdown(_inline_ai_context_html(lang), unsafe_allow_html=True)
    st.markdown(
        '<div class="inline-ai-evidence-note">'
        '<b>' + html.escape("Local companion" if is_en else "本地伴随助手") + '</b>'
        '<p>' + html.escape(
            "Use this dock for screen-aware help and fast workspace actions. Open the full Copilot for a complete guided study."
            if is_en else
            "这个 dock 用于当前页面解释和快速驱动工作区；完整引导式研究请打开完整 Copilot。"
        ) + '</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    action_cols = st.columns(3, gap="small")
    with action_cols[0]:
        if st.button(
            "Full Copilot" if is_en else "完整 Copilot",
            key="_floating_ai_full_copilot",
            use_container_width=True,
            icon=":material/auto_awesome:",
        ):
            st.session_state["llm_enabled"] = True
            st.session_state["_llm_toggle"] = True
            st.session_state["_active_main_page"] = "assistant"
            st.session_state["_floating_ai_open"] = False
            st.session_state["_scroll_to_top"] = True
            st.rerun()
    with action_cols[1]:
        if st.button(
            "Demo review" if is_en else "演示审阅",
            key="_floating_ai_demo_review",
            use_container_width=True,
            icon=":material/preview:",
        ):
            _apply_chat_workflow_action("demo_review")
            st.rerun()
    with action_cols[2]:
        if st.button(
            "Agent setup" if is_en else "Agent 设置",
            key="_floating_ai_agent_setup",
            use_container_width=True,
            icon=":material/assignment:",
        ):
            _prepare_research_agent_handoff_from_ai(st.session_state)
            st.rerun()


def render_floating_chat_dock():
    """Render a fixed bottom-right floating AI chat dock."""
    _init_chat_state()
    lang = st.session_state.get("language", "en")
    if "_floating_ai_open" not in st.session_state:
        st.session_state["_floating_ai_open"] = False
    if "_floating_ai_size" not in st.session_state:
        st.session_state["_floating_ai_size"] = "m"
    if st.session_state.get("_ai_pending_question"):
        st.session_state["_floating_ai_open"] = True

    # Defaults match easyicu design/page-ai-chat.jsx: 420 × 780 panel.
    size_presets = {
        "s": {
            "panel_width": "clamp(320px, 28vw, 380px)",
            "panel_max_height": "min(72vh, 640px)",
            "history_height": 280,
        },
        "m": {
            "panel_width": "420px",
            "panel_max_height": "min(86vh, 780px)",
            "history_height": 420,
        },
        "l": {
            "panel_width": "clamp(460px, 38vw, 640px)",
            "panel_max_height": "min(92vh, 960px)",
            "history_height": 560,
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

        /* Notification badge on the launcher button — design token bad. */
        div.st-key-floating_ai_launcher .ai-notif-badge {
            position: absolute;
            top: -4px; right: 2px;
            min-width: 18px; height: 18px;
            border-radius: 9px;
            background: var(--bad, #b54f3e);
            color: #fff;
            font-size: 10.5px;
            font-weight: 500;
            display: flex; align-items: center; justify-content: center;
            padding: 0 5px;
            box-shadow: var(--sh-1, 0 1px 3px rgba(14,17,22,0.15));
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
            border: 1px solid var(--ink);
            background: var(--ink) !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            font-size: clamp(1.05rem, 0.8vw + 0.9rem, 1.35rem);
            box-shadow: var(--sh-3, 0 18px 44px rgba(14,17,22,0.18));
        }


        /* Floating panel — page-ai-chat.jsx 420×780 surface card. */
        div.st-key-floating_ai_panel {
            right: clamp(12px, 1.25vw, 20px);
            bottom: clamp(12px, 1.25vw, 20px);
            width: min(var(--easyicu-ai-panel-width), calc(100vw - 24px));
            max-width: calc(100vw - 24px);
            max-height: var(--easyicu-ai-panel-max-height);
            border-radius: 14px;
            border: 1px solid var(--hair-2);
            background: var(--surface);
            box-shadow: var(--sh-3, 0 24px 58px rgba(14, 17, 22, 0.16));
            backdrop-filter: none;
            padding: 0;
            overflow-x: hidden;
            overflow-y: auto;
            animation: easyicuPanelSlideIn 0.28s cubic-bezier(0.22, 1, 0.36, 1) both;
            color: var(--ink);
        }

        div.st-key-floating_ai_panel .floating-ai-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 14px 8px;
            border-bottom: 1px solid var(--hair);
        }

        div.st-key-floating_ai_panel .floating-ai-title-row {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        div.st-key-floating_ai_panel .floating-ai-avatar {
            width: 28px;
            height: 28px;
            border-radius: 8px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-ink);
            background: var(--accent-soft);
            border: 1px solid var(--accent-border);
            box-shadow: none;
            font-size: 14px;
            flex: 0 0 auto;
        }

        div.st-key-floating_ai_panel .floating-ai-title {
            font-size: 13.5px;
            font-weight: 500;
            color: var(--ink);
            letter-spacing: 0;
        }

        div.st-key-floating_ai_panel .floating-ai-subtitle {
            font-size: 10.5px;
            color: var(--ink-4);
            margin-top: 2px;
            line-height: 1.35;
        }

        div.st-key-floating_ai_panel .floating-ai-welcome {
            background: var(--accent-soft);
            border: 1px solid var(--accent-border);
            border-left: 3px solid var(--accent);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 10px;
            box-shadow: none;
            color: var(--accent-ink);
        }

        div.st-key-floating_ai_panel .floating-ai-welcome-title {
            font-size: 12.5px;
            font-weight: 500;
            color: var(--ink);
            margin-bottom: 4px;
        }

        div.st-key-floating_ai_panel .floating-ai-welcome-subtitle {
            font-size: 11.5px;
            line-height: 1.5;
            color: var(--ink-2);
            margin-bottom: 8px;
        }

        div.st-key-floating_ai_panel .floating-ai-sample {
            display: grid;
            gap: 6px;
            margin: 6px 0 8px;
        }

        /* page-ai-chat user bubble: ink fill, white text */
        div.st-key-floating_ai_panel .floating-ai-user-bubble {
            justify-self: end;
            max-width: 85%;
            color: #fff;
            background: var(--ink);
            border: 1px solid var(--ink);
            border-radius: 10px;
            padding: 8px 12px;
            font-size: 12.5px;
            line-height: 1.55;
            box-shadow: none;
        }

        /* page-ai-chat agent bubble: surface, hair border */
        div.st-key-floating_ai_panel .floating-ai-answer-card {
            color: var(--ink);
            background: var(--surface);
            border: 1px solid var(--hair);
            border-radius: 10px;
            padding: 8px 12px;
            font-size: 12.5px;
            line-height: 1.55;
            box-shadow: none;
        }

        div.st-key-floating_ai_panel .floating-ai-recommendation {
            display: grid;
            gap: 2px;
            color: var(--ink-2);
            background: var(--surface-2);
            border: 1px solid var(--hair);
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 11px;
        }

        div.st-key-floating_ai_panel .floating-ai-recommendation span {
            color: var(--ink-4);
            font-size: 9.5px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        div.st-key-floating_ai_panel .floating-ai-recommendation strong {
            color: var(--ink);
            font-weight: 500;
            line-height: 1.42;
        }

        div.st-key-floating_ai_panel .inline-ai-context-card,
        div.st-key-floating_ai_panel .inline-ai-evidence-note {
            border-radius: 10px;
            border: 1px solid var(--hair);
            background: var(--surface-2);
            padding: 10px 12px;
            margin: 8px 8px 10px;
        }

        div.st-key-floating_ai_panel .inline-ai-context-card h3,
        div.st-key-floating_ai_panel .inline-ai-evidence-note b {
            display: block;
            margin: 0;
            color: var(--ink);
            font-size: 12.5px;
            font-weight: 600;
            line-height: 1.35;
        }

        div.st-key-floating_ai_panel .inline-ai-context-card p,
        div.st-key-floating_ai_panel .inline-ai-evidence-note p {
            margin: 4px 0 0;
            color: var(--ink-3);
            font-size: 11.5px;
            line-height: 1.45;
        }

        div.st-key-floating_ai_panel .inline-ai-section-label {
            color: var(--ink-4);
            font-family: var(--font-mono);
            font-size: 9.5px;
            font-weight: 500;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }

        div.st-key-floating_ai_panel .inline-ai-tag-row {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 8px;
        }

        div.st-key-floating_ai_panel .inline-ai-tag-row span {
            border-radius: 999px;
            border: 1px solid var(--hair);
            background: var(--surface);
            color: var(--ink-3);
            font-size: 10.5px;
            padding: 2px 7px;
        }

        div.st-key-floating_ai_panel .floating-ai-welcome-hint {
            font-size: 10px;
            color: var(--accent-ink);
            font-weight: 500;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        div.st-key-floating_ai_panel div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.2rem 0.3rem 0.4rem 0.3rem;
        }

        div.st-key-floating_ai_panel [data-testid="stChatMessage"] {
            margin-bottom: 0.45rem;
        }

        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] {
            border-radius: 10px;
            padding: 8px 12px;
            background: var(--surface);
            border: 1px solid var(--hair);
            box-shadow: none;
        }

        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] p,
        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] li {
            font-size: 12.5px;
            line-height: 1.55;
            color: var(--ink);
        }

        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] ul,
        div.st-key-floating_ai_panel [data-testid="stChatMessageContent"] ol {
            padding-left: 1.15rem;
            margin-top: 0.45rem;
            margin-bottom: 0.2rem;
        }

        div.st-key-floating_ai_panel .stButton > button {
            border-radius: 6px;
            min-height: 28px;
            font-size: 11.5px;
            padding: 4px 10px;
            border: 1px solid var(--hair-2) !important;
            color: var(--ink) !important;
            -webkit-text-fill-color: var(--ink) !important;
            background: var(--surface) !important;
            box-shadow: none !important;
        }
        div.st-key-floating_ai_panel .stButton > button:hover {
            background: var(--surface-2) !important;
            border-color: var(--hair-3) !important;
        }

        div.st-key-floating_ai_panel .stButton > button[kind="primary"],
        div.st-key-floating_ai_panel .stButton > button[data-testid="stBaseButton-primary"],
        div.st-key-floating_ai_panel form button,
        div.st-key-floating_ai_panel div[data-testid="stFormSubmitButton"] button {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            border: 1px solid var(--ink) !important;
            background: var(--ink) !important;
            box-shadow: none !important;
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
            if st.button(
                "",
                key="_floating_ai_open_btn",
                help="Open Research Copilot dock" if lang == "en" else "打开研究 Copilot dock",
                icon=":material/smart_toy:",
            ):
                st.session_state["_floating_ai_open"] = True
                st.rerun()

    if st.session_state.get("_floating_ai_open", False):
        with st.container(key="floating_ai_panel"):
            dock_title = "Research Copilot" if lang == "en" else "研究 Copilot"
            dock_subtitle = (
                "Grounded in your current page and selections."
                if lang == "en" else
                "基于当前页面和已选配置给出建议。"
            )
            provider_key = coerce_public_provider(
                st.session_state.get("llm_provider", public_default_provider_key())
            )
            provider_label = public_provider_defaults(provider_key)[0] or provider_key
            llm_configured = _is_configured()
            if llm_configured:
                status_label = f"{provider_label} · online"
                status_color = "var(--ok)"
            else:
                status_label = "local Copilot" if lang == "en" else "本地 Copilot"
                status_color = "var(--accent)"
            status_pill = (
                f'<span class="eu-pill mono" '
                f'style="font-size:10px;padding:2px 7px;height:18px;'
                f'background:var(--surface);color:{status_color};'
                f'border:1px solid var(--hair-2)">'
                f'<span class="dot" style="background:{status_color}"></span>'
                f'{html.escape(status_label)}</span>'
            )
            header_cols = st.columns([4.8, 0.8, 0.8, 0.8, 1, 1])
            with header_cols[0]:
                st.markdown(
                    f'''
                    <div class="floating-ai-header">
                        <div class="floating-ai-title-row">
                            <span class="floating-ai-avatar" aria-label="AI assistant">✦</span>
                            <div>
                                <div class="floating-ai-title">{dock_title} {status_pill}</div>
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

            _render_floating_copilot_context_actions(lang)
            if not _is_configured():
                st.caption(
                    "Local Copilot is available. Configure a provider only for open-ended external model calls."
                    if lang == "en" else
                    "本地 Copilot 可直接使用。只有开放式外部模型调用才需要配置服务商/API Key。"
                )
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
#### What is Research Copilot?
A built-in conversational helper that knows EasyICU inside-out. It can:
- Start from your study goal, then map it to the right EasyICU workflow
- Explain which cohort filters, feature modules, and scores fit your task
- List supported databases, concepts, and scoring systems
- Help interpret extraction results (SOFA, Sepsis-3, missingness, etc.)

**Getting started:**
1. Toggle **Enable Research Copilot** in the sidebar
2. Choose a provider and enter your API key or token
3. Start by describing your task, e.g. "I want to build a sepsis early-warning cohort."
4. If you already know the disease cohort you want, ask directly for AKI / Sepsis / ventilation / ICD-based filtering.
""")
    else:
        st.markdown("""\
#### 研究 Copilot 是什么？
内置的对话助手，熟知 EasyICU 的所有功能。它可以：
- 从你的研究目标出发，反推最合适的 EasyICU 工作流
- 解释适合该任务的队列筛选、特征模块和临床评分
- 列出支持的数据库、概念和评分系统
- 帮助解读提取结果（SOFA、Sepsis-3、缺失率等）

**快速开始：**
1. 在侧边栏开启 **启用研究 Copilot**
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
        st.error("No model specified." if lang == "en" else "未指定模型名称。")
        return

    status_placeholder = st.empty()
    answer_placeholder = st.empty()

    try:
        status_placeholder.info(
            "Thinking..." if lang == "en" else "正在思考..."
        )
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        status_placeholder.info(
            "Generating response..." if lang == "en" else "正在生成回答..."
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
                "Hugging Face requires your own token. "
                "Please create an HF token and paste it here."
                if lang == "en" else
                "Hugging Face 需要你自己提供 token。"
                "请创建 HF token 后再填写。"
            )
        else:
            msg = ("Authentication failed — please check your API Key."
                   if lang == "en" else "认证失败 — 请检查 API Key 是否正确。")
    elif (
        "429" in err_str
        or "rate" in err_lower
        or "temporarily rate-limited" in err_lower
        or "provider returned error" in err_lower
        or "retry shortly" in err_lower
    ):
        msg = (
            "The current hosted model is being rate-limited upstream. Please retry shortly, or switch the hosted default model to a more stable free model."
            if lang == "en" else
            "当前托管模型被上游限流了。请稍后重试，或把 hosted 默认模型切换到更稳定的免费模型。"
        )
    elif "socksio" in err_lower or "using socks proxy" in err_lower:
        msg = (
            "Proxy configuration error — the app detected a SOCKS proxy from the environment. "
            "EasyICU now ignores system proxy variables by default. Please retry. "
            "If you explicitly need a SOCKS proxy, install `httpx[socks]`."
            if lang == "en" else
            "代理配置异常 — 应用检测到了环境变量中的 SOCKS 代理。"
            "EasyICU 现在默认忽略系统代理变量，请重试。"
            "如果你确实需要 SOCKS 代理，请安装 `httpx[socks]`。"
        )
    elif "model" in err_lower or "404" in err_str:
        msg = ("Model not found — please verify the model name."
               if lang == "en" else "模型未找到 — 请确认模型名称是否正确。")
    elif "connect" in err_lower or "timeout" in err_lower:
        msg = ("Connection error — check the API Base URL and your network."
               if lang == "en" else "连接失败 — 请检查 API Base URL 和网络连接。")
    else:
        msg = ("API error: " if lang == "en" else "API 调用出错: ") + err_str
    if render:
        st.error(msg)
    return msg
