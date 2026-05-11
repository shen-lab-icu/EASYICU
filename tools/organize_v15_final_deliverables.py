#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def copy_file(src: Path, dst: Path, rows: list[dict[str, Any]], category: str, note: str, required: bool = True) -> bool:
    if not src.exists():
        rows.append({
            "category": category,
            "deliverable_file": str(dst),
            "source_file": str(src),
            "exists": False,
            "size_bytes": "",
            "note": note,
        })
        if required:
            return False
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    rows.append({
        "category": category,
        "deliverable_file": str(dst),
        "source_file": str(src),
        "exists": True,
        "size_bytes": dst.stat().st_size,
        "note": note,
    })
    return True


def copy_tree_files(src_dir: Path, dst_dir: Path, rows: list[dict[str, Any]], category: str, note: str, suffixes: set[str] | None = None) -> int:
    if not src_dir.exists():
        rows.append({
            "category": category,
            "deliverable_file": str(dst_dir),
            "source_file": str(src_dir),
            "exists": False,
            "size_bytes": "",
            "note": f"missing source directory: {note}",
        })
        return 0
    count = 0
    for src in sorted(src_dir.rglob("*")):
        if not src.is_file():
            continue
        if suffixes is not None and src.suffix.lower() not in suffixes:
            continue
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        copy_file(src, dst, rows, category, note, required=False)
        count += 1
    return count


def count_rows(path: Path) -> int:
    return len(read_csv(path))


def build_start_here(out_dir: Path, run_root: Path, audit_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# EasyICU v15 最终文件整理包",
        "",
        f"生成时间：`{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## 先看这 4 个文件",
        "",
        "1. `00_START_HERE/README_START_HERE.md`：这个文件，告诉你从哪里开始。",
        "2. `04_writing/manuscript_draft/draft_manuscript_en.md`：英文论文草稿。",
        "3. `04_writing/one_page_summary_plain.md`：一页版中文总结。",
        "4. `05_for_collaborators/collaborator_note_plain.md`：发给合作者的简短说明。",
        "",
        "## 这个整理包做了什么？",
        "",
        "它把 final audit、curated publication、manuscript evidence 里的关键文件复制到一个更清楚的目录里。",
        "",
        "它没有移动原始实验目录，也没有删除任何文件。原始证据仍然留在原来的 run directory。",
        "",
        "## 最重要的原始目录",
        "",
        f"- 原始实验根目录：`{run_root}`",
        f"- final audit 目录：`{audit_dir}`",
        f"- curated publication 目录：`{audit_dir / 'curated_publication'}`",
        f"- manuscript evidence 目录：`{audit_dir / 'manuscript_evidence'}`",
        "",
        "## 整理后目录怎么读？",
        "",
        "- `00_START_HERE/`：入口说明和阅读顺序。",
        "- `01_experiment_audit/`：最终实验状态、审计表、修复记录。",
        "- `02_publication_figures/`：论文可用图，包括主图、补充图、重画图。",
        "- `03_publication_tables/`：论文可用表，包括主表和补充表。",
        "- `04_writing/`：论文草稿、摘要、中文解释、caption、检查清单。",
        "- `05_for_collaborators/`：适合发给合作者快速审阅的文件。",
        "- `06_provenance/`：来源索引，说明每个整理文件来自哪里。",
        "- `07_original_locations/`：原始目录位置说明，不复制大体积 run 文件。",
        "",
        "## 写论文时最重要的提醒",
        "",
        "- 可以说：60 个实验单元都在执行和审计框架下完成。",
        "- 必须说：有自动修复和兜底逻辑。",
        "- 不要说：模型完全自主完成全部科学分析。",
        "- 不要说：乳酸、血管活性药物、年龄等变量导致死亡。这里多数是相关关系。",
        "- 重画图只是为了展示清楚，不是新实验。",
        "",
        "## 文件索引",
        "",
        "完整文件索引见：`06_provenance/DELIVERABLE_FILE_INDEX.csv`。",
        "",
        f"本整理包共登记 `{len(rows)}` 个复制或索引记录。",
    ]
    write_text(out_dir / "00_START_HERE" / "README_START_HERE.md", lines)


def build_reading_order(out_dir: Path) -> None:
    lines = [
        "# 推荐阅读顺序",
        "",
        "## 如果你只想快速了解结果",
        "",
        "1. `04_writing/one_page_summary_plain.md`",
        "2. `04_writing/manuscript_draft/abstract_draft_en.md`",
        "3. `05_for_collaborators/collaborator_note_plain.md`",
        "",
        "## 如果你要开始写论文",
        "",
        "1. `04_writing/manuscript_draft/draft_manuscript_en.md`",
        "2. `04_writing/manuscript_draft/draft_manuscript_zh_explained.md`",
        "3. `04_writing/captions/main_figure_captions_plain.md`",
        "4. `04_writing/captions/supplement_figure_captions_plain.md`",
        "5. `04_writing/manuscript_draft/figure_table_placement_plan_plain.md`",
        "",
        "## 如果你要检查实验是否可靠",
        "",
        "1. `01_experiment_audit/FINAL_STATUS.md`",
        "2. `01_experiment_audit/matrix_status.csv`",
        "3. `01_experiment_audit/metric_sanity_audit.csv`",
        "4. `01_experiment_audit/repair_audit.csv`",
        "5. `06_provenance/figure_source_map.csv`",
        "6. `06_provenance/table_source_map.csv`",
        "",
        "## 如果你要发给合作者",
        "",
        "建议发这些：",
        "",
        "- `05_for_collaborators/collaborator_note_plain.md`",
        "- `04_writing/one_page_summary_plain.md`",
        "- `04_writing/manuscript_draft/abstract_draft_en.md`",
        "- `02_publication_figures/main/`",
        "- `02_publication_figures/rebuilt/`",
        "- `03_publication_tables/main/`",
    ]
    write_text(out_dir / "00_START_HERE" / "READING_ORDER.md", lines)


def build_original_locations(out_dir: Path, run_root: Path, audit_dir: Path) -> None:
    lines = [
        "# 原始文件位置说明",
        "",
        "这个整理包没有复制完整原始 run 目录，因为原始实验输出体积大，而且复制后容易造成多个版本混淆。",
        "",
        "## 原始实验根目录",
        "",
        f"`{run_root}`",
        "",
        "这里保存完整 15 × 4 实验输出。",
        "",
        "## final audit 目录",
        "",
        f"`{audit_dir}`",
        "",
        "这里保存 frozen audit、审计表、图像清单、最终写作材料包。",
        "",
        "## curated publication 目录",
        "",
        f"`{audit_dir / 'curated_publication'}`",
        "",
        "这里保存论文展示层图表。",
        "",
        "## manuscript evidence 目录",
        "",
        f"`{audit_dir / 'manuscript_evidence'}`",
        "",
        "这里保存论文草稿、白话说明、caption 和检查清单。",
        "",
        "## 如何追踪整理文件来源？",
        "",
        "看 `06_provenance/DELIVERABLE_FILE_INDEX.csv`。每一行都有 source_file。",
    ]
    write_text(out_dir / "07_original_locations" / "ORIGINAL_LOCATIONS.md", lines)


def build_package(run_root: Path, audit_dir: Path, out_dir: Path) -> None:
    curated = audit_dir / "curated_publication"
    manuscript = audit_dir / "manuscript_evidence"
    rows: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_files = [
        "FINAL_STATUS.md",
        "PAPER_TABLES_README.md",
        "figure_review_notes.md",
        "repair_burden_summary.md",
        "metric_interpretation_notes.md",
        "matrix_status.csv",
        "repair_audit.csv",
        "metric_sanity_audit.csv",
        "artifact_inventory.csv",
        "figure_inventory.csv",
        "paper_task_success_by_arm.csv",
        "paper_task_metric_summary.csv",
        "paper_repair_burden.csv",
        "figure_review_shortlist.csv",
        "final_audit_summary.json",
    ]
    for name in audit_files:
        copy_file(audit_dir / name, out_dir / "01_experiment_audit" / name, rows, "experiment_audit", "final audit key file", required=False)
    copy_file(run_root / "README_FINAL.md", out_dir / "01_experiment_audit" / "README_FINAL.md", rows, "experiment_audit", "final run README", required=False)
    copy_tree_files(audit_dir / "snapshot", out_dir / "01_experiment_audit" / "snapshot", rows, "experiment_snapshot", "frozen matrix/summary/log snapshot", None)

    copy_tree_files(curated / "figures" / "main", out_dir / "02_publication_figures" / "main", rows, "publication_main_figures", "main candidate figures", {".png", ".svg", ".pdf"})
    copy_tree_files(curated / "figures" / "supplement", out_dir / "02_publication_figures" / "supplement", rows, "publication_supplement_figures", "supplement candidate figures", {".png", ".svg", ".pdf"})
    copy_tree_files(curated / "figures" / "rebuilt", out_dir / "02_publication_figures" / "rebuilt", rows, "publication_rebuilt_figures", "rebuilt presentation figures", {".png", ".svg", ".pdf"})
    copy_tree_files(curated / "tables" / "main", out_dir / "03_publication_tables" / "main", rows, "publication_main_tables", "main paper tables", {".csv", ".md"})
    copy_tree_files(curated / "tables" / "supplement", out_dir / "03_publication_tables" / "supplement", rows, "publication_supplement_tables", "supplement tables", {".csv", ".md"})

    writing_roots = [
        "README.md",
        "one_page_summary_plain.md",
        "glossary_plain.md",
        "collaborator_note_plain.md",
        "package_summary.json",
    ]
    for name in writing_roots:
        copy_file(manuscript / name, out_dir / "04_writing" / name, rows, "writing_entry", "plain-language writing entry file", required=False)
    for sub in ["methods", "results", "captions", "checklist", "submission_package", "manuscript_draft"]:
        copy_tree_files(manuscript / sub, out_dir / "04_writing" / sub, rows, f"writing_{sub}", f"manuscript evidence {sub}", None)

    collaborator_files = [
        (manuscript / "collaborator_note_plain.md", "collaborator_note_plain.md"),
        (manuscript / "one_page_summary_plain.md", "one_page_summary_plain.md"),
        (manuscript / "manuscript_draft" / "abstract_draft_en.md", "abstract_draft_en.md"),
        (manuscript / "manuscript_draft" / "figure_table_placement_plan_plain.md", "figure_table_placement_plan_plain.md"),
        (manuscript / "checklist" / "manuscript_claim_checklist.csv", "manuscript_claim_checklist.csv"),
    ]
    for src, name in collaborator_files:
        copy_file(src, out_dir / "05_for_collaborators" / name, rows, "collaborator_packet", "small packet for collaborator review", required=False)

    copy_file(curated / "source_maps" / "figure_source_map.csv", out_dir / "06_provenance" / "figure_source_map.csv", rows, "provenance", "curated figure source map", required=False)
    copy_file(curated / "source_maps" / "table_source_map.csv", out_dir / "06_provenance" / "table_source_map.csv", rows, "provenance", "curated table source map", required=False)
    copy_file(manuscript / "package_summary.json", out_dir / "06_provenance" / "manuscript_package_summary.json", rows, "provenance", "manuscript evidence summary", required=False)

    build_start_here(out_dir, run_root, audit_dir, rows)
    build_reading_order(out_dir)
    build_original_locations(out_dir, run_root, audit_dir)

    index_path = out_dir / "06_provenance" / "DELIVERABLE_FILE_INDEX.csv"
    write_csv(index_path, rows, ["category", "deliverable_file", "source_file", "exists", "size_bytes", "note"])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "audit_dir": str(audit_dir),
        "out_dir": str(out_dir),
        "deliverable_records": len(rows),
        "matrix_rows": count_rows(audit_dir / "matrix_status.csv"),
        "repair_rows": count_rows(audit_dir / "repair_audit.csv"),
        "figure_source_rows": count_rows(curated / "source_maps" / "figure_source_map.csv"),
        "table_source_rows": count_rows(curated / "source_maps" / "table_source_map.csv"),
    }
    (out_dir / "06_provenance" / "DELIVERABLE_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else audit_dir / "final_deliverables_20260510_60clean"
    build_package(run_root, audit_dir, out_dir)
    print(out_dir)
    print(out_dir / "00_START_HERE" / "README_START_HERE.md")
    print(out_dir / "06_provenance" / "DELIVERABLE_FILE_INDEX.csv")
    print(out_dir / "06_provenance" / "DELIVERABLE_SUMMARY.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
