"""
EasyICU LLM Chat Assistant Module.

Provides an embedded conversational AI assistant that helps users
understand EasyICU features, interpret extraction results, and
answer ICU data analysis questions.

Supported providers:
    - HuggingFace (free credits, token required)
  - OpenAI, DeepSeek, Anthropic, OpenRouter, Together AI, Groq,
    SiliconFlow, and any custom OpenAI-compatible endpoint.

All API credentials are stored in session state only — never persisted.
"""
import ast
import os
import re
from functools import lru_cache
from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# System prompt — enriched with EasyICU documentation & concept catalogue
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an intelligent assistant embedded in **EasyICU**, an interactive platform \
for clinical data extraction and exploration across multiple public ICU databases.

## Platform Overview
EasyICU is a Python toolkit (v1.0) that provides:
- Unified access to **6 public ICU databases**: MIMIC-IV (miiv), MIMIC-III (mimic), \
eICU-CRD (eicu), AmsterdamUMCdb (aumc), HiRID (hirid), SICdb (sic)
- Automated extraction of **167 standardized clinical concepts** across 19 feature modules
- A no-code Streamlit web interface for cohort construction, feature selection, \
quality review, and cohort comparison
- High-performance computing optimised for 16 GB RAM machines
- Export in Parquet, CSV, and XLSX formats; all parameters exportable as JSON for reproducibility

## Workflow (4 Steps)
1. **Data Source** — choose database & path (or Demo mode with simulated data)
2. **Cohort Selection** — filter by age, sex, ICU LOS, mortality, first-ICU-stay
3. **Select Features** — pick from 19 modules (167 concepts); supports SOFA-1, SOFA-2, \
Sepsis-3, KDIGO-AKI, circulatory failure, etc.
4. **Export Data** — batch export to disk; streaming architecture, subprocess memory isolation

## Main Web Areas
- **Tutorial** — workflow guide, usage examples, and the in-app data dictionary
- **Quick Visualization** — load extracted data and inspect trends, missingness, and distributions
- **Cohort Analysis** — compare groups and review downstream cohort summaries
- **AI Assistant** — guided help for navigation, feature planning, troubleshooting, and evidence lookup

## Feature Modules & Concepts (19 modules, 167 concepts)
- **Vital Signs** (7): hr, map, sbp, dbp, temp, spo2, resp
- **Respiratory** (14): pafi, safi, fio2, supp_o2, vent_ind, vent_start, vent_end, \
o2sat, sao2, mech_vent, ett_gcs, ecmo, ecmo_indication, adv_resp
- **Ventilator Parameters** (12): peep, tidal_vol, tidal_vol_set, pip, plateau_pres, \
mean_airway_pres, minute_vol, vent_rate, etco2, compliance, driving_pres, ps
- **Blood Gas** (9): be, cai, hbco, lact, methb, pco2, ph, po2, tco2
- **Chemistry** (21): alb, alp, alt, ast, bicar, bili, bili_dir, bun, ca, ck, ckmb, \
cl, crea, crp, glu, k, mg, na, phos, tnt, tri
- **Hematology** (20): bnd, basos, eos, esr, fgn, hba1c, hct, hgb, inr_pt, lymph, \
mch, mchc, mcv, neut, plt, pt, ptt, rbc, rdw, wbc
- **Vasopressors** (17): norepi_rate/dur/equiv/60, epi_rate/dur/60, \
dopa_rate/dur/60, dobu_rate/dur/60, adh_rate, phn_rate, vaso_ind, other_vaso
- **Medications** (4): abx, cort, dex, ins
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
- Prioritize helping users use the EasyICU web interface and workflows. Code-level explanations are secondary unless the user explicitly asks for implementation details.
- When a user asks where something is in the web app, answer with the relevant page, step, or in-app action first. Do not default to repo files.
- When the user asks about EasyICU implementation, prefer the exact EasyICU concept names and outputs over generic medical summaries.
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


def _hosted_base_url() -> str:
    """Return the hosted relay URL exposed to end users."""
    return os.getenv("EASYICU_HOSTED_BASE_URL", "http://47.241.42.236/v1").strip()


def _repo_blob_base() -> str:
    """Return the GitHub blob base used for clickable file links."""
    return os.getenv(
        "EASYICU_REPO_BLOB_BASE",
        "https://github.com/shen-lab-icu/easyicu/blob/main",
    ).rstrip("/")


def _default_provider_key() -> str:
    """Prefer hosted mode when a hosted relay URL is configured."""
    return "easyicu_hosted" if _hosted_base_url() else "openrouter"

# ---------------------------------------------------------------------------
# Provider registry: (display, default_base_url, default_model, needs_key, description_en, description_zh)
# ---------------------------------------------------------------------------
PROVIDERS = {
    "easyicu_hosted": (
        "EasyICU Hosted",
        _hosted_base_url(),
        "hosted-default",
        False,
        "Use the EasyICU managed relay. No user API key required.",
        "使用 EasyICU 托管代理，无需用户自己填写 API Key。",
    ),
    "huggingface_free": (
        "🆓 HuggingFace (Free credits)",
        "https://router.huggingface.co/v1",
        "deepseek-ai/DeepSeek-R1:fastest",
        True,
        "Free credits are available, but a Hugging Face token is still required.",
        "有免费额度，但仍然需要 Hugging Face token。",
    ),
    "openai": (
        "OpenAI",
        "https://api.openai.com/v1",
        "gpt-4o",
        True,
        "GPT-4o, GPT-4o-mini, o1, etc.",
        "GPT-4o、GPT-4o-mini、o1 等。",
    ),
    "deepseek": (
        "DeepSeek",
        "https://api.deepseek.com",
        "deepseek-chat",
        True,
        "DeepSeek-V3, DeepSeek-R1. Very affordable.",
        "DeepSeek-V3、DeepSeek-R1，价格极低。",
    ),
    "anthropic": (
        "Anthropic",
        "https://api.anthropic.com/v1",
        "claude-sonnet-4-20250514",
        True,
        "Claude Sonnet, Opus, Haiku.",
        "Claude Sonnet、Opus、Haiku。",
    ),
    "openrouter": (
        "OpenRouter",
        "https://openrouter.ai/api/v1",
        "deepseek/deepseek-chat-v3-0324:free",
        True,
        "Aggregator with free & paid models. Get key at openrouter.ai",
        "模型聚合平台，有免费和付费模型。在 openrouter.ai 获取 Key。",
    ),
    "together": (
        "Together AI",
        "https://api.together.xyz/v1",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        True,
        "Llama, Mistral, Qwen. Free tier available (signup).",
        "Llama、Mistral、Qwen。注册即有免费额度。",
    ),
    "groq": (
        "Groq",
        "https://api.groq.com/openai/v1",
        "llama-3.3-70b-versatile",
        True,
        "Ultra-fast inference. Free tier with rate limits.",
        "超低延迟推理。免费套餐有速率限制。",
    ),
    "siliconflow": (
        "SiliconFlow (硅基流动)",
        "https://api.siliconflow.cn/v1",
        "deepseek-ai/DeepSeek-V3",
        True,
        "China-based, DeepSeek/Qwen. Free tier available.",
        "国内平台，DeepSeek/Qwen 等模型，注册赠送额度。",
    ),
    "custom": (
        "⚙️ Custom / Compatible",
        "",
        "",
        True,
        "Any OpenAI-compatible endpoint.",
        "任意 OpenAI 兼容接口。",
    ),
}

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _init_chat_state():
    """Ensure all chat-related session keys exist."""
    default_provider = _default_provider_key()
    defaults = {
        "llm_enabled": True,
        "llm_provider": default_provider,
        "llm_api_key": "",
        "llm_model": "",
        "llm_base_url": "",
        "llm_messages": [],
        "llm_configured": False,
        "llm_last_tool_events": [],
        "llm_last_verification": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
    combined = f"{(prompt or '').lower()}\n{(answer or '').lower()}"
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

    if any(key in combined for key in ["字典", "数据字典", "dictionary", "feature list", "concept list"]):
        add_nav("home_dict", "📖 Open Data Dictionary", "📖 打开数据字典")

    if any(key in combined for key in ["tutorial", "教程", "step", "步骤", "how do i", "怎么做", "workflow", "流程", "guide", "使用"]):
        add_nav("tutorial", "📚 Open Tutorial", "📚 打开教程")

    if any(key in combined for key in [
        "quick visualization", "快速可视化", "load data", "加载数据", "visualization",
        "visualize", "visualise", "plot", "可视化", "图表", "数据分析", "分析我的数据",
    ]):
        add_nav("viz", "📊 Open Quick Visualization", "📊 前往快速可视化")

    if any(key in combined for key in ["cohort", "队列", "compare", "comparison", "dashboard", "仪表板"]):
        add_nav("cohort", "🔬 Open Cohort Analysis", "🔬 前往队列分析")

    if any(key in combined for key in ["export", "导出"]):
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
    """Render streaming text manually so it can be replaced after verification."""
    chunks = []
    for token in _token_generator(stream):
        chunks.append(token)
        placeholder.markdown("".join(chunks) + "▌")
    text = "".join(chunks).strip()
    placeholder.markdown(text)
    return text


def _parse_verification_report(text: str) -> dict[str, object]:
    """Parse verifier output into a structured result."""
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
            or PROVIDERS.get(st.session_state.get("llm_provider", "custom"), PROVIDERS["custom"])[2],
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
    return PROVIDERS.get(provider, PROVIDERS["custom"])[3]


def _is_configured() -> bool:
    """Return True when the provider is ready to use."""
    provider = st.session_state.get("llm_provider", "openrouter")
    if not _needs_api_key(provider):
        default_url = PROVIDERS.get(provider, PROVIDERS["custom"])[1]
        return bool((st.session_state.get("llm_base_url", "") or default_url).strip())
    return bool(st.session_state.get("llm_api_key", "").strip())


def _get_client():
    """Build and return an OpenAI-compatible client, or *None* on error."""
    try:
        from openai import OpenAI
        import httpx
    except ImportError:
        return None

    provider = st.session_state.get("llm_provider", "openrouter")
    api_key = st.session_state.get("llm_api_key", "").strip()
    base_url = st.session_state.get("llm_base_url", "").strip() or None

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
    lang = st.session_state.get("language", "en")

    label = "🤖 AI Assistant" if lang == "en" else "🤖 AI 助手"
    with st.expander(label, expanded=False):
        # On/off toggle
        enabled = st.toggle(
            "Enable AI Assistant" if lang == "en" else "启用 AI 助手",
            value=st.session_state.llm_enabled,
            key="_llm_toggle",
        )
        st.session_state.llm_enabled = enabled
        if not enabled:
            hint = ("Enable the toggle, then switch to the 🤖 AI Assistant tab."
                    if lang == "en"
                    else "开启开关后，切换到 🤖 AI 助手 标签页即可使用。")
            st.caption(hint)
            return

        # Provider
        provider_keys = list(PROVIDERS.keys())
        idx = provider_keys.index(st.session_state.llm_provider) \
            if st.session_state.llm_provider in provider_keys else 0
        provider = st.selectbox(
            "Provider" if lang == "en" else "服务商",
            options=provider_keys,
            index=idx,
            format_func=lambda k: PROVIDERS[k][0],
            key="_llm_provider_sel",
        )
        st.session_state.llm_provider = provider

        # Provider description
        p_info = PROVIDERS[provider]
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

        jump_label = "🤖 Open AI Assistant" if lang == "en" else "🤖 打开 AI 助手"
        if st.button(jump_label, use_container_width=True, key="_goto_ai_assistant"):
            st.session_state["_scroll_to_tab"] = "ai_assistant"
            st.rerun()

        # Clear history button
        if st.session_state.llm_messages:
            if st.button("🗑️ " + ("Clear Chat" if lang == 'en' else "清空对话"),
                         key="_llm_clear"):
                st.session_state.llm_messages = []
                st.rerun()


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
    provider_name = PROVIDERS.get(
        st.session_state.llm_provider, PROVIDERS["custom"])[0]
    model_name = (st.session_state.get("llm_model", "").strip()
                  or PROVIDERS.get(st.session_state.llm_provider,
                                   PROVIDERS["custom"])[2]
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

    if st.session_state.get("llm_last_verification"):
        verification = st.session_state.llm_last_verification
        with st.expander("🔎 " + ("Last verification" if lang == "en" else "上次校验"), expanded=False):
            status = verification.get("status", "uncertain")
            label_map = {
                "pass": "✅ Pass" if lang == "en" else "✅ 通过",
                "corrected": "🛠️ Corrected" if lang == "en" else "🛠️ 已纠正",
                "uncertain": "⚠️ Uncertain" if lang == "en" else "⚠️ 不确定",
            }
            st.markdown(label_map.get(status, status))
            for issue in verification.get("issues", [])[:5]:
                st.markdown(f"- {issue}")

    # ---- Render message history -----------------------------------------------
    history_container = st.container(height=680, border=True)
    with history_container:
        for msg_idx, msg in enumerate(st.session_state.llm_messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("actions"):
                    _render_nav_actions(msg["actions"], key_prefix=f"_llm_action_{msg_idx}")

    # ---- Chat input -----------------------------------------------------------
    placeholder = ("Ask a question about EasyICU …"
                    if lang == "en" else "输入关于 EasyICU 的问题 …")
    if prompt := st.chat_input(placeholder, key="_llm_chat_input"):
        # Append user message
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
            st.session_state.llm_messages.append(
                {
                    "role": "assistant",
                    "content": instant_reply,
                    "actions": _suggest_ui_actions(prompt, instant_reply, lang),
                }
            )
            with history_container:
                with st.chat_message("assistant"):
                    st.markdown(instant_reply)
                    _render_nav_actions(
                        _suggest_ui_actions(prompt, instant_reply, lang),
                        key_prefix="_llm_action_instant",
                    )
            return

        prep_placeholder = st.empty()
        prep_placeholder.info(
            "🛠️ Preparing tools..." if lang == "en" else "🛠️ 正在准备工具..."
        )

        # Build agent payload with local tools / external evidence
        messages, tool_events = _compose_agent_messages(prompt)
        st.session_state.llm_last_tool_events = tool_events
        prep_placeholder.empty()

        # Call LLM with streaming
        with history_container:
            with st.chat_message("assistant"):
                _stream_response(messages, lang)


# ---------------------------------------------------------------------------
# Intro / tips helpers
# ---------------------------------------------------------------------------

def _render_intro(lang: str):
    """Show a brief feature overview when the assistant is disabled."""
    if lang == "en":
        st.markdown("""\
#### What is the AI Assistant?
A built-in conversational helper that knows EasyICU inside-out. It can:
- 📖 Explain any feature, concept, or workflow step
- 🗄️ List supported databases, concepts, and scoring systems
- 📊 Help interpret extraction results (SOFA, missingness, etc.)
- 💡 Answer general ICU data analysis questions

**Getting started:**
1. Toggle **Enable AI Assistant** in the sidebar
2. Choose a provider and enter your API key or token
3. Start chatting here
""")
    else:
        st.markdown("""\
#### AI 助手是什么？
内置的对话助手，熟知 EasyICU 的所有功能。它可以：
- 📖 解释任何功能、概念或工作流步骤
- 🗄️ 列出支持的数据库、概念和评分系统
- 📊 帮助解读提取结果（SOFA、缺失率等）
- 💡 回答 ICU 数据分析相关的通识问题

**快速开始：**
1. 在侧边栏开启 **启用 AI 助手**
2. 选择服务商并填写对应的 API Key / Token
3. 在此标签页开始对话
""")


def _render_tips(lang: str):
    if lang == "en":
        st.markdown("""\
- **Onboarding**: "I want to extract SOFA-2 from MIMIC-IV. What exact steps should I follow in the web UI?"
- **Feature planning**: "For septic shock research, which EasyICU concepts should I export besides SOFA and vasopressors?"
- **Cross-database mapping**: "Which respiratory concepts are available across miiv, mimic, eicu, aumc, hirid, and sic?"
- **Troubleshooting**: "My exported data shows high missingness for fio2. What are the most likely causes and checks?"
- **Interpretation**: "How should I interpret `sep3_sofa2`, `susp_inf`, and `sofa2` together?"
- **Code-aware help**: "Where is export implemented in app.py?" / "How does `load_concepts` work?"
- **Evidence-backed answers**: "Explain Sepsis-3 with PubMed sources and relate it to EasyICU outputs."
- **Python workflow**: "Show me a minimal Python example to load pafi, sofa2, and sep3_sofa2."
""")
    else:
        st.markdown("""\
- **新手引导**: "我想从 MIMIC-IV 提取 SOFA-2，网页端具体点哪里、按什么顺序做？"
- **选特征建议**: "如果我要做脓毒症休克研究，除了 SOFA 和升压药，还建议导出哪些概念？"
- **跨库对照**: "miiv、mimic、eicu、aumc、hirid、sic 里哪些呼吸相关概念都能取到？"
- **排错诊断**: "我导出的 fio2 缺失率很高，最可能是什么原因，应该检查哪几步？"
- **结果解读**: "`sep3_sofa2`、`susp_inf` 和 `sofa2` 应该怎么一起解释？"
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

    provider = st.session_state.get("llm_provider", "openrouter")
    model = (st.session_state.get("llm_model", "").strip()
             or PROVIDERS.get(provider, PROVIDERS["custom"])[2])
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
        draft_response = _stream_text(stream, answer_placeholder)
        status_placeholder.info(
            "🔎 Verifying answer..." if lang == "en" else "🔎 正在校验回答..."
        )
        verification = _verify_response(client, messages, draft_response, lang)
        final_response = verification.get("corrected_answer") or draft_response
        final_response = _append_quick_links(
            prompt=st.session_state.llm_messages[-1]["content"],
            answer=final_response,
            lang=lang,
        )
        answer_placeholder.markdown(final_response)
        st.session_state.llm_last_verification = verification
        response_actions = _suggest_ui_actions(
            st.session_state.llm_messages[-1]["content"],
            final_response,
            lang,
        )
        _render_nav_actions(response_actions, key_prefix="_llm_action_live")

        verify_status = verification.get("status", "uncertain")
        verify_event = {
            "tool": "answer_verifier",
            "status": "ok" if verify_status in ("pass", "corrected") else "error",
            "detail": f"status={verify_status}",
        }
        st.session_state.llm_last_tool_events = [
            *st.session_state.get("llm_last_tool_events", []),
            verify_event,
        ]
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
        _handle_api_error(exc, lang)


def _token_generator(stream):
    """Yield content tokens from an OpenAI-style streaming response."""
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                yield token


def _handle_api_error(exc: Exception, lang: str):
    """Display a user-friendly error for common API failures."""
    err_str = str(exc)
    if "authentication" in err_str.lower() or "401" in err_str:
        provider = st.session_state.get("llm_provider", "custom")
        if provider == "huggingface_free":
            msg = (
                "❌ Hugging Face requires a token even for the free credits tier. "
                "Please create an HF token and paste it here."
                if lang == "en" else
                "❌ Hugging Face 即使是免费额度也需要 token。"
                "请创建 HF token 后再填写。"
            )
        else:
            msg = ("❌ Authentication failed — please check your API Key."
                   if lang == "en" else "❌ 认证失败 — 请检查 API Key 是否正确。")
    elif "socksio" in err_str.lower() or "using socks proxy" in err_str.lower():
        msg = (
            "🌐 Proxy configuration error — the app detected a SOCKS proxy from the environment. "
            "EasyICU now ignores system proxy variables by default. Please retry. "
            "If you explicitly need a SOCKS proxy, install `httpx[socks]`."
            if lang == "en" else
            "🌐 代理配置异常 — 应用检测到了环境变量中的 SOCKS 代理。"
            "EasyICU 现在默认忽略系统代理变量，请重试。"
            "如果你确实需要 SOCKS 代理，请安装 `httpx[socks]`。"
        )
    elif "rate" in err_str.lower() or "429" in err_str:
        msg = ("⏳ Rate limit reached — please wait a moment and try again."
               if lang == "en" else "⏳ 请求频率超限 — 请稍后再试。")
    elif "model" in err_str.lower() or "404" in err_str:
        msg = ("⚠️ Model not found — please verify the model name."
               if lang == "en" else "⚠️ 模型未找到 — 请确认模型名称是否正确。")
    elif "connect" in err_str.lower() or "timeout" in err_str.lower():
        msg = ("🌐 Connection error — check the API Base URL and your network."
               if lang == "en" else "🌐 连接失败 — 请检查 API Base URL 和网络连接。")
    else:
        msg = ("❌ API error: " if lang == "en" else "❌ API 调用出错: ") + err_str
    st.error(msg)
