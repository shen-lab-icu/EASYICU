"""What each allow-listed run file is for, and where a run download files it.

The run review and the download bundle read one rule for the run's report.
A revision is the current report and the first report becomes history.
Without a revision the first report is the manuscript only when the gate's
``manuscript_ready`` check passed; otherwise it is a report draft.

The bundle keeps the reviewed bytes of every allow-listed file and adds
reader copies decoded from them: each embedded gallery figure as an image
file and each aggregate result table as CSV, with a bilingual README.
"""

from __future__ import annotations

import base64
import binascii
import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "BUNDLE_FOLDERS",
    "RunFileGuideEntry",
    "bundle_files",
    "first_report_state",
    "run_file_guide",
]

# Folder, English heading, Chinese heading -- in reading order.
BUNDLE_FOLDERS: Tuple[Tuple[str, str, str], ...] = (
    ("manuscript", "Manuscript", "稿件"),
    ("results", "Results", "结果"),
    ("plan", "Plan and literature", "计划与文献"),
    ("checks", "Checks and review", "核验与审阅"),
    ("provenance", "Provenance", "溯源"),
)

_Text = Tuple[str, str, str, str, str]  # folder, title en/zh, purpose en/zh

# Titles and purposes follow the run review's own labels.
_FILES: Dict[str, _Text] = {
    "manuscript_revision.pdf": (
        "manuscript", "Current report revision (PDF)", "当前报告修订（PDF）",
        "The current report, typeset from locked, evidence-bound results.",
        "由已锁定、证据绑定的结果排版的当前报告。",
    ),
    "manuscript_provenance.json": (
        "manuscript", "Evidence-bound manuscript reader", "证据绑定论文阅读器",
        "Each bound number in the text with its exact field, step, and registered code/data lineage.",
        "正文中每个绑定数字对应的准确字段、分析步骤及已登记的代码/数据链路。",
    ),
    "result_tables.json": (
        "results", "Research result tables", "科研结果表",
        "Bounded aggregate table previews from registered Research Agent evidence.",
        "来自 Research Agent 已登记证据的有界聚合表格预览。",
    ),
    "figure_gallery.json": (
        "results", "Figure gallery", "图件画廊",
        "Task-specific figures rendered from this completed run.",
        "这道问题已渲染出的任务特异图件。",
    ),
    "cohort_summary.json": (
        "results", "Cohort summary", "队列摘要",
        "Denominator, cohort basis, and outcome availability.",
        "分母、队列依据与结局可用性。",
    ),
    "table1_summary.json": (
        "results", "Table 1 summary", "Table 1 摘要",
        "Baseline characteristics of the analysed cohort.",
        "分析队列的基线特征。",
    ),
    "missingness_audit.json": (
        "results", "Missingness audit", "缺失审计",
        "Missing values by variable in the analysed cohort.",
        "分析队列中各变量的缺失情况。",
    ),
    "roc_curve.json": (
        "results", "ROC curve", "ROC 曲线",
        "Discrimination of the prediction model.",
        "预测模型的区分度。",
    ),
    "calibration_curve.json": (
        "results", "Calibration curve", "校准曲线",
        "Agreement between predicted and observed risk.",
        "预测风险与观察风险的一致性。",
    ),
    "agent_plan.json": (
        "plan", "Agent plan", "Agent 计划",
        "The step-by-step analysis plan used by the Agent.",
        "Agent 执行时使用的分步分析计划。",
    ),
    "scientific_plan_review.json": (
        "plan", "Scientific plan review", "科学计划审阅",
        "Digest-bound multi-dimensional review before the plan can be approved.",
        "计划批准前的摘要绑定多维科学审阅。",
    ),
    "literature_evidence.json": (
        "plan", "Literature evidence", "文献证据",
        "Search provenance, article metadata, and exact plan-step citation bindings.",
        "检索溯源、文章元数据以及计划步骤的精确文献绑定。",
    ),
    "quality_gate.json": (
        "checks", "Evidence check", "证据核验",
        "Automated checks explaining why the run remains analysis-only.",
        "自动核验结果，说明为何仍保持 analysis-only。",
    ),
    "scientific_readiness.json": (
        "checks", "Scientific readiness", "科学就绪情况",
        "Which scientific checks passed and which remain open for this run.",
        "本次运行哪些科学核查已通过、哪些仍未关闭。",
    ),
    "human_signoff.json": (
        "checks", "Human sign-off", "人工签署",
        "The local reviewer's sign-off; it grants no publication authority.",
        "本地审阅者的签署；不授予发表权限。",
    ),
    "benchmark_scorecard.json": (
        "checks", "Evaluation scorecard", "评估记分卡",
        "Plan, code, evidence binding, and safety scores for this research run.",
        "本次研究运行的计划、代码、证据绑定与安全评分。",
    ),
    "system_validation_report.pdf": (
        "checks", "System validation dossier (PDF)", "系统验证报告（PDF）",
        "Engineering validation dossier; not a clinical manuscript.",
        "工程验证报告；不是临床论文。",
    ),
    "system_validation_report.html": (
        "checks", "System validation dossier (HTML)", "系统验证报告（HTML）",
        "Engineering validation dossier; not a clinical manuscript.",
        "工程验证报告；不是临床论文。",
    ),
    "system_validation_report.json": (
        "checks", "System validation dossier", "系统验证报告",
        "Source-bound engineering validation; explicitly not a clinical manuscript.",
        "源绑定的工程验证报告；明确不是临床论文。",
    ),
    "run_context.json": (
        "provenance", "Run context", "运行上下文",
        "Question, cohort, source run, and local project metadata.",
        "研究问题、队列、原始运行与本地项目元数据。",
    ),
    "evidence_ledger.json": (
        "provenance", "Evidence ledger", "证据账本",
        "Artifact hashes, evidence ids, and privacy-audit status.",
        "产物哈希、证据 ID 与隐私审计状态。",
    ),
    "manuscript_draft.json": (
        "provenance", "Locked manuscript draft", "锁定论文草稿",
        "Locked claims and evidence ids; not a reportable manuscript.",
        "锁定论断及其证据 ID；不是可报告论文草稿。",
    ),
    "manuscript_pdf_receipt.json": (
        "provenance", "Report PDF receipt", "报告 PDF 回执",
        "Digest receipt binding the report PDF to its source.",
        "将报告 PDF 与其来源绑定的摘要回执。",
    ),
    "system_validation_report_receipt.json": (
        "provenance", "System validation receipt", "系统验证回执",
        "Digest receipt for the engineering-only report and rendered document.",
        "仅限工程用途的报告及渲染文档摘要回执。",
    ),
    "source_run_manifest.json": (
        "provenance", "Source run manifest", "原始运行清单",
        "Original completed run provenance and import manifest.",
        "原始完成运行的溯源与导入清单。",
    ),
    "workflow_graph.json": (
        "provenance", "Workflow graph", "工作流图谱",
        "Agent steps and handoffs from question to evidence review.",
        "从研究问题到证据审阅的 Agent 步骤与交接。",
    ),
}

# The first report under each state of the report rule.
_FIRST_REPORT: Dict[str, _Text] = {
    "history": (
        "provenance", "Original run report (PDF)", "原运行报告（历史 PDF）",
        "The report as this run first produced it; kept for provenance.",
        "本次运行最初生成的报告，留作溯源。",
    ),
    "manuscript": (
        "manuscript", "Manuscript PDF", "稿件 PDF",
        "The manuscript this run produced, for download.",
        "本次运行生成的稿件，可下载。",
    ),
    "draft": (
        "manuscript", "Report draft (PDF)", "报告草稿（PDF）",
        "The report this run produced; its manuscript review is still open.",
        "本次运行生成的报告草稿；稿件审阅尚未通过。",
    ),
}

# The first report's sources travel with it.
_FIRST_REPORT_SOURCES: Dict[str, Tuple[str, str, str, str]] = {
    "manuscript_scaffold.tex": (
        "Report LaTeX source", "报告 LaTeX 源文件",
        "The typeset source of the first report.", "首份报告的排版源文件。",
    ),
    "manuscript_scaffold.bib": (
        "Report bibliography (BibTeX)", "报告参考文献（BibTeX）",
        "References cited by the first report.", "首份报告引用的参考文献。",
    ),
}

_ORDER = (
    "manuscript_revision.pdf", "manuscript_scaffold.pdf", "manuscript_provenance.json",
    "manuscript_scaffold.tex", "manuscript_scaffold.bib",
    "result_tables.json", "figure_gallery.json", "cohort_summary.json", "table1_summary.json",
    "missingness_audit.json", "roc_curve.json", "calibration_curve.json",
    "agent_plan.json", "scientific_plan_review.json", "literature_evidence.json",
    "quality_gate.json", "scientific_readiness.json", "human_signoff.json",
    "benchmark_scorecard.json", "system_validation_report.pdf", "system_validation_report.html",
    "system_validation_report.json",
)

# Gate check ids read in both languages; an unknown id reads as its words.
_CHECK_NAMES: Dict[str, Tuple[str, str]] = {
    "execution_complete": ("execution complete", "执行完成"),
    "analysis_validated": ("analysis validated", "分析已验证"),
    "evidence_complete": ("evidence complete", "证据完整"),
    "numeric_verified": ("numbers verified", "数值已核对"),
    "manuscript_ready": ("manuscript ready", "稿件就绪"),
    "publication_ready": ("publication ready", "可发表"),
    "paper_authorized": ("submission authorized", "已授权投稿"),
    "human_signoff": ("human sign-off", "人工签署"),
}


@dataclass(frozen=True)
class RunFileGuideEntry:
    """One allow-listed run file, read by what it is for."""

    name: str
    folder: str
    title_en: str
    title_zh: str
    purpose_en: str
    purpose_zh: str

    @property
    def bundle_path(self) -> str:
        return f"{self.folder}/{self.name}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "folder": self.folder,
            "bundle_path": self.bundle_path,
            "title": {"en": self.title_en, "zh": self.title_zh},
            "purpose": {"en": self.purpose_en, "zh": self.purpose_zh},
        }


def first_report_state(
    names: Iterable[str], *, gate_checks: Sequence[Mapping[str, Any]]
) -> str:
    """``history``, ``manuscript`` or ``draft`` for the run's first report."""

    if "manuscript_revision.pdf" in set(names):
        return "history"
    passed = any(
        isinstance(check, Mapping)
        and check.get("id") == "manuscript_ready"
        and check.get("passed") is True
        for check in gate_checks
    )
    return "manuscript" if passed else "draft"


def run_file_guide(
    names: Iterable[str], *, gate_checks: Sequence[Mapping[str, Any]]
) -> Tuple[RunFileGuideEntry, ...]:
    """Guide entries for the present files, in reading order."""

    present = list(dict.fromkeys(str(name) for name in names if name))
    state = first_report_state(present, gate_checks=gate_checks)
    first_folder = _FIRST_REPORT[state][0]
    entries: List[RunFileGuideEntry] = []
    for name in present:
        if name == "manuscript_scaffold.pdf":
            text = _FIRST_REPORT[state]
        elif name in _FIRST_REPORT_SOURCES:
            text = (first_folder, *_FIRST_REPORT_SOURCES[name])
        else:
            text = _FILES.get(name) or ("provenance", name, name, "", "")
        entries.append(RunFileGuideEntry(name, *text))
    folder_rank = {folder: index for index, (folder, _, _) in enumerate(BUNDLE_FOLDERS)}
    name_rank = {name: index for index, name in enumerate(_ORDER)}
    return tuple(sorted(
        entries,
        key=lambda entry: (
            folder_rank[entry.folder], name_rank.get(entry.name, len(name_rank)), entry.name,
        ),
    ))


def _json_object(raw: Optional[bytes]) -> Mapping[str, Any]:
    if raw is None:
        return {}
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _unique(path: str, taken: set) -> str:
    stem, dot, suffix = path.rpartition(".")
    candidate, index = path, 2
    while candidate in taken:
        candidate = f"{stem}_{index}{dot}{suffix}" if dot else f"{path}_{index}"
        index += 1
    taken.add(candidate)
    return candidate


def _safe_stem(value: str, fallback: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return stem[:80] or fallback


_DATA_URL = re.compile(r"data:image/(png|jpeg|svg\+xml);base64,([A-Za-z0-9+/=]+)")
_IMAGE_SUFFIX = {"png": "png", "jpeg": "jpg", "svg+xml": "svg"}


def _image_bytes(data_url: Any) -> Optional[Tuple[str, bytes]]:
    match = _DATA_URL.fullmatch(str(data_url or ""))
    if match is None:
        return None
    kind, encoded = match.groups()
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        return None
    signatures = {"png": (b"\x89PNG\r\n\x1a\n",), "jpeg": (b"\xff\xd8\xff",), "svg+xml": (b"<svg", b"<?xml")}
    if not data.lstrip().startswith(signatures[kind]):
        return None
    return _IMAGE_SUFFIX[kind], data


def _gallery_figures(gallery: Mapping[str, Any], taken: set) -> List[Tuple[str, bytes, str, str]]:
    figures = []
    for index, figure in enumerate(gallery.get("figures") or (), start=1):
        if not isinstance(figure, Mapping):
            continue
        image = _image_bytes(figure.get("data_url"))
        if image is None:
            continue
        suffix, data = image
        stem = _safe_stem(PurePosixPath(str(figure.get("name") or "")).stem, f"figure_{index}")
        path = _unique(f"results/figures/{stem}.{suffix}", taken)
        figures.append((path, data, _figure_title(figure.get("label"), stem), str(figure.get("caption") or "")))
    return figures


def _sentence_case(words: str) -> str:
    words = " ".join(words.replace("_", " ").split())
    return words[:1].upper() + words[1:]


def _figure_title(label: Any, stem: str) -> str:
    # Gallery labels may carry the product kind ("figure:cohort flow").
    text = str(label or "").strip()
    text = text.split(":", 1)[1] if text.lower().startswith("figure:") else text
    return _sentence_case(text or stem)


def _table_name(table: Mapping[str, Any], index: int) -> str:
    name = PurePosixPath(str(table.get("name") or "")).name
    name = name.split("__", 1)[1] if "__" in name else name
    return _safe_stem(name.rsplit(".", 1)[0] if "." in name else name, f"table_{index}")


def _result_tables(tables: Mapping[str, Any], taken: set) -> List[Tuple[str, bytes, str, str]]:
    written, seen = [], set()
    for index, table in enumerate(tables.get("tables") or (), start=1):
        if not isinstance(table, Mapping):
            continue
        headers, rows = table.get("headers"), table.get("rows")
        if not isinstance(headers, list) or not isinstance(rows, list):
            continue
        content = json.dumps([headers, rows], sort_keys=True, default=str)
        if content in seen:
            continue  # the same aggregate copied beside a figure
        seen.add(content)
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["" if cell is None else cell for cell in headers])
        for row in rows:
            if isinstance(row, list):
                writer.writerow(["" if cell is None else cell for cell in row])
        name = _table_name(table, index)
        path = _unique(f"results/tables/{name}.csv", taken)
        notes = []
        if table.get("preview_truncated"):
            notes.append("Preview rows only · 仅含预览行")
        if table.get("preview_columns_truncated"):
            notes.append("Some columns omitted; result_tables.json names the source · 部分列未包含，来源见 result_tables.json")
        written.append((path, buffer.getvalue().encode("utf-8"), _sentence_case(name), "; ".join(notes)))
    return written


def _check_words(check_id: str) -> Tuple[str, str]:
    words = check_id.replace("_", " ")
    return _CHECK_NAMES.get(check_id, (words, words))


def _readme(
    guide: Sequence[RunFileGuideEntry],
    *,
    context: Mapping[str, Any],
    gate_checks: Sequence[Mapping[str, Any]],
    figures: Sequence[Tuple[str, bytes, str, str]],
    tables: Sequence[Tuple[str, bytes, str, str]],
) -> bytes:
    lines = ["# EasyICU run files · 运行文件", ""]
    source = context.get("source") if isinstance(context.get("source"), Mapping) else {}
    for label, value in (
        ("Run · 运行", context.get("run_id")),
        ("Data source · 数据来源", source.get("label") or source.get("database")),
        ("Question · 研究问题", context.get("question")),
    ):
        if value:
            lines.append(f"- {label}: {' '.join(str(value).split())}")
    open_checks = [
        _check_words(str(check.get("id")))
        for check in gate_checks
        if isinstance(check, Mapping) and check.get("id") and check.get("passed") is not True
    ]
    if open_checks:
        lines.append("- Open checks · 未通过的核验: " + "; ".join(f"{en} · {zh}" for en, zh in open_checks))
    elif gate_checks:
        lines.append("- Open checks · 未通过的核验: none · 无")
    lines += [
        "",
        "Analysis only: these are the run's allow-listed review files; they grant no publication authority.",
        "仅供分析：以下是本次运行白名单内的审阅文件，不授予发表权限。",
    ]
    for folder, heading_en, heading_zh in BUNDLE_FOLDERS:
        rows = [entry for entry in guide if entry.folder == folder]
        extras = []
        if folder == "results":
            extras += [(path, title, "图件", caption) for path, _, title, caption in figures]
            extras += [(path, title, "结果表 CSV", note) for path, _, title, note in tables]
        if not rows and not extras:
            continue
        lines += ["", f"## {folder}/ · {heading_en} · {heading_zh}", ""]
        for path, title_en, title_zh, note in extras:
            lines.append(f"- `{path.removeprefix(folder + '/')}` — {title_en} · {title_zh}")
            if note:
                lines.append(f"  {' '.join(note.split())}")
        for entry in rows:
            lines.append(f"- `{entry.name}` — {entry.title_en} · {entry.title_zh}")
            if entry.purpose_en:
                lines.append(f"  {entry.purpose_en}")
            if entry.purpose_zh:
                lines.append(f"  {entry.purpose_zh}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def bundle_files(contents: Mapping[str, bytes]) -> List[Tuple[str, bytes]]:
    """Folder the reviewed bytes of each file and add the reader copies.

    ``contents`` holds each allow-listed file's bytes after its privacy scan.
    """

    gate = _json_object(contents.get("quality_gate.json")).get("gate")
    gate = gate if isinstance(gate, Mapping) else {}
    checks = [check for check in gate.get("checks") or () if isinstance(check, Mapping)]
    guide = run_file_guide(contents, gate_checks=checks)
    taken = {entry.bundle_path for entry in guide} | {"README.md"}
    figures = _gallery_figures(_json_object(contents.get("figure_gallery.json")), taken)
    tables = _result_tables(_json_object(contents.get("result_tables.json")), taken)
    readme = _readme(
        guide,
        context=_json_object(contents.get("run_context.json")),
        gate_checks=checks,
        figures=figures,
        tables=tables,
    )
    return [
        ("README.md", readme),
        *[(entry.bundle_path, contents[entry.name]) for entry in guide],
        *[(path, data) for path, data, _, _ in figures],
        *[(path, data) for path, data, _, _ in tables],
    ]
