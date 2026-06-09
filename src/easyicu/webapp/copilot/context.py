"""Research Copilot — codebase / PubMed context + floating-copilot local answers.

Extracted from `llm_chat.py` (Phase-6 split, 7th batch). Backs the floating
copilot's offline Q&A: repo-file introspection (outline/snippet/identifier
extraction, project index), quick GitHub file links, PubMed E-utilities lookup,
the inline AI context payload, and the deterministic local-answer / instant-reply
/ starter-prompt fallbacks. Pure helpers — **no Streamlit `st` access** — so the
test suite's `monkeypatch.setattr(llm_chat, "st", ...)` contract does not apply.

`_repo_root` deliberately stays in `llm_chat.py`: it resolves the repo root from
`Path(__file__).parents[N]`, which is position-sensitive, so moving it one
directory deeper would shift the anchor. It (plus `PROJECT_CONTEXT_FILES` /
`PROJECT_LINK_FILES`) is lazy-imported inside the using functions, so this module
never imports `llm_chat` at load time — no cycle. `llm_chat.py` re-imports every
name below, so all call sites keep working.
"""
from __future__ import annotations

import ast
import os
import re
from collections.abc import MutableMapping
from functools import lru_cache
from pathlib import Path

import requests


def _repo_blob_base() -> str:
    """Return the GitHub blob base used for clickable file links."""
    return os.getenv(
        "EASYICU_REPO_BLOB_BASE",
        "https://github.com/shen-lab-icu/easyicu/blob/main",
    ).rstrip("/")


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
    from easyicu.webapp.llm_chat import PROJECT_CONTEXT_FILES, _repo_root  # lazy
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
    from easyicu.webapp.llm_chat import PROJECT_CONTEXT_FILES  # lazy
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
    from easyicu.webapp.llm_chat import PROJECT_LINK_FILES  # lazy
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


def _local_copilot_fallback_reply(prompt: str, lang: str) -> str:
    if lang == "en":
        return (
            "I can handle this locally as Research Copilot. Describe a study goal, say `run the whole demo`, "
            "ask me to explore a review-derived idea, or open demo review, real data setup, or Research Agent "
            "handoff. Enable an external provider only when you need open-ended evidence lookup or long-form explanation."
        )
    return (
        "我可以先用本地 Research Copilot 逻辑处理：描述一个研究目标，直接说“跑完整演示”，"
        "或让我探索文献 idea、打开演示审阅、真实数据配置、Research Agent 交接。只有需要开放式证据检索或长篇解释时，才需要启用外部模型。"
    )


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


def _starter_prompts(lang: str) -> list[str]:
    if lang == "en":
        return [
            "Explore a review-derived ICU research idea, then stop at the registry gate.",
            "Start a guided ICU outcome study and ask me for the endpoint first.",
            "Start a cross-database study and ask me for cohort, outcome, and databases one by one.",
            "Walk me through a data-quality audit before choosing cohort or concepts.",
            "Walk me through the real data source step and explain the prepared path first.",
        ]
    return [
        "探索一个来自综述的 ICU 研究 idea，并停在 registry 人工关口。",
        "开始一个 ICU 结局研究向导，先问我要研究的 endpoint。",
        "开始一个跨数据库研究，逐步问我队列、结局和数据库。",
        "先带我体验数据质量审计，再选择队列或概念。",
        "先带我完成真实数据源步骤，并解释 prepared 路径。",
    ]


def _floating_copilot_answer_text(answer_id: str, lang: str) -> str:
    """Local, design-aligned dock answers that do not require an external model."""
    is_en = lang == "en"
    answers_en = {
        "how": (
            "EasyICU moves a study through four local stages: extract -> review -> analyze -> draft. "
            "You can drive each classic workspace yourself, or let Research Copilot assemble the same "
            "state by chat. Patient rows stay on this machine."
        ),
        "extract": (
            "Data Extraction is the four-step gate: choose a source, define the cohort, select concept "
            "modules, then export reproducible frames plus a manifest. Each later page reads that same "
            "frozen setup."
        ),
        "export": (
            "An extraction export is concept data plus a manifest. It is not a manuscript bundle; tables, "
            "figures, and drafts come from Research Agent runs and stay gated by evidence checks."
        ),
        "patient": (
            "Patient Review checks the loaded cohort before analysis: selected concepts, patient-level "
            "tables, time series, quality flags, and denominators. It is the human review step before a run."
        ),
        "tabs": (
            "Patient Review has four live tabs: Data Tables for stay-level rows, Time Series for hourly "
            "trajectories, Patient Overview for one selected stay, and Data Quality for coverage, ranges, "
            "and missingness."
        ),
        "quality": (
            "Data-quality flags mark sparse, missing, duplicated, or out-of-range concepts before they can "
            "bias a cohort review or downstream model. Treat a flag as a denominator check, not as a finding."
        ),
        "cohort": (
            "Cohort Statistics turns the reviewed data into denominators, group contrast, coverage, cohort "
            "profile, and SOFA reclassification panels. Research Agent drafts only after these checks are clear."
        ),
        "sofa": (
            "SOFA reclassification shows how patients move between severity bands when the score is recomputed. "
            "Treat it as a sensitivity and consistency check, not as a standalone new finding."
        ),
        "contrast": (
            "The default cohort contrast is Survived vs Deceased. You can switch to age groups, sex, length of "
            "stay, or Sepsis vs non-sepsis; every contrast should keep denominators visible."
        ),
        "overlap": (
            "Cross-DB Benchmark compares standardized concepts across selected ICU databases. The availability "
            "matrix shows present, partial, and missing concepts before distribution deltas are interpreted."
        ),
        "gate": (
            "The draft is locked by design. A claim can only be written after denominators, coverage, artifacts, "
            "and reviewer sign-off all trace back to the local evidence ledger."
        ),
        "states": (
            "Workspace States documents the reusable loading, empty, no-data, error, blocked, and success "
            "surfaces so every page communicates where the workflow is and what action is possible."
        ),
        "privacy": (
            "EasyICU is local-first. Extraction, review, and analysis use local files; external providers are "
            "optional and only for open-ended model calls. Patient rows are not sent by this dock."
        ),
        "idea": (
            "Stay in chat and tell me the clinical area or database constraint. I will suggest a few executable "
            "directions, then wait for you to pick or edit one before configuring data, cohort, and modules."
        ),
    }
    answers_zh = {
        "how": (
            "EasyICU 把研究推进为四个本地阶段：提取 -> 审阅 -> 分析 -> 草稿。你可以自己操作经典工作区，"
            "也可以让 Research Copilot 通过聊天组装同一套状态；患者行数据留在本机。"
        ),
        "extract": (
            "Data Extraction 是四步闸门：选择数据源、定义队列、选择概念模块、导出可复现数据帧和 manifest。"
            "后续页面都会读取这套冻结配置。"
        ),
        "export": (
            "提取导出包含概念数据和 manifest，不是 manuscript bundle。表格、图件和草稿来自 Research Agent "
            "运行，并继续受证据检查闸门约束。"
        ),
        "patient": (
            "Patient Review 用于分析前人工核对：已加载概念、患者级表格、时间序列、质量标记和分母。"
            "这是 Research Agent 运行前的审阅步骤。"
        ),
        "tabs": (
            "Patient Review 有四个实时标签页：Data Tables 显示 stay 级行数据，Time Series 显示小时级轨迹，"
            "Patient Overview 聚焦单个住院，Data Quality 汇总覆盖率、范围和缺失情况。"
        ),
        "quality": (
            "数据质量标记会在队列审阅或下游建模前提示稀疏、缺失、重复或越界的概念。"
            "它应作为分母检查，而不是直接当成研究发现。"
        ),
        "cohort": (
            "Cohort Statistics 把已审阅数据整理成分母、组间对比、覆盖率、队列画像和 SOFA 重分类面板。"
            "这些检查通过前，Research Agent 不应写草稿论断。"
        ),
        "sofa": (
            "SOFA 重分类展示重新计算评分时患者如何在严重程度分层之间移动。它是敏感性和一致性检查，"
            "不是单独的新发现。"
        ),
        "contrast": (
            "默认队列对比是存活 vs 死亡。你可以切换到年龄组、性别、住院时长或 Sepsis vs 非 Sepsis；"
            "每个对比都应该保留清楚的分母。"
        ),
        "overlap": (
            "Cross-DB Benchmark 比较多个 ICU 数据库中的标准化概念。Availability matrix 会先显示 present、"
            "partial、missing，再解释分布差异。"
        ),
        "gate": (
            "草稿默认锁定是有意设计。只有分母、覆盖率、产物和人工 sign-off 都能追溯到本地 evidence ledger 后，"
            "才允许写出论断。"
        ),
        "states": (
            "Workspace States 记录 loading、empty、no-data、error、blocked、success 等可复用状态，"
            "让每个页面都清楚表达当前进度和可执行动作。"
        ),
        "privacy": (
            "EasyICU 是 local-first。提取、审阅和分析使用本地文件；外部 provider 只用于可选开放式模型调用。"
            "这个 dock 不发送患者行数据。"
        ),
        "idea": (
            "留在聊天里告诉我临床方向或数据库约束。我会先给几个可执行研究方向，等你选择或修改后，"
            "再继续配置数据源、队列和模块。"
        ),
    }
    library = answers_en if is_en else answers_zh
    return library.get(answer_id, library["how"])


def _append_floating_copilot_local_answer(
    state: MutableMapping[str, object],
    *,
    label: str,
    answer_id: str,
    lang: str,
) -> None:
    """Write a local dock answer into the shared Copilot chat history."""
    messages = state.setdefault("llm_messages", [])
    if isinstance(messages, list):
        messages.append({"role": "user", "content": label})
        messages.append({
            "role": "assistant",
            "content": _floating_copilot_answer_text(answer_id, lang),
            "actions": [],
        })
    state["_floating_ai_open"] = True
    state.pop("_ai_pending_question", None)


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
