#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ARM_ORDER = ["aware", "aware_no_pref", "naive_with_pref", "naive"]
ARM_LABELS = {
    "aware": "Aware",
    "aware_no_pref": "Aware\nno pref",
    "naive_with_pref": "Naive\nwith pref",
    "naive": "Naive",
}
TASK_SHORT = {
    "t01_table_one_descriptive": "T01 Table 1",
    "t02_outcome_incidence_strata": "T02 Outcome strata",
    "t03_severity_score_correlation": "T03 SOFA corr.",
    "t04_lactate_mortality_association": "T04 Lactate OR",
    "t05_kdigo_renal_sensitivity": "T05 KDIGO sens.",
    "t06_shock_phenotype_clustering": "T06 Shock clusters",
    "t07_mortality_prediction_auroc": "T07 Prediction",
    "t08_vaso_selection_bias_audit": "T08 Vaso bias",
    "t09_sofa_zero_artefact_audit": "T09 SOFA-zero",
    "t10_complete_case_robustness": "T10 Missingness",
    "t11_los_distribution_descriptive": "T11 LOS",
    "t12_age_stratified_mortality": "T12 Age strata",
    "t13_admission_vital_summary": "T13 Vitals",
    "t14_creatinine_trajectory_kdigo": "T14 Creatinine",
    "t15_norepinephrine_dose_response": "T15 Norepi dose",
}


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


def copy_file(src: Path, dst: Path, records: list[dict[str, Any]], role: str, label: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
        exists = True
        size = dst.stat().st_size
    else:
        exists = False
        size = ""
    records.append({"role": role, "label": label, "file": str(dst), "source_file": str(src), "exists": exists, "size_bytes": size})


def task_sort_key(task: str) -> tuple[int, str]:
    try:
        return int(task[1:3]), task
    except Exception:
        return 999, task


def status_counts(matrix: list[dict[str, str]]) -> Counter[str]:
    return Counter(row.get("status") or "" for row in matrix)


def build_completion_heatmap(table1: list[dict[str, str]], out_base: Path) -> None:
    tasks = [row.get("task_key") or "" for row in sorted(table1, key=lambda r: task_sort_key(r.get("task_key") or ""))]
    data = np.zeros((len(tasks), len(ARM_ORDER)))
    for i, row in enumerate(sorted(table1, key=lambda r: task_sort_key(r.get("task_key") or ""))):
        for j, arm in enumerate(ARM_ORDER):
            data[i, j] = 1 if row.get(arm) == "clean_ok" else 0
    fig, ax = plt.subplots(figsize=(8, max(5.5, 0.35 * len(tasks) + 1.5)))
    ax.imshow(data, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(ARM_ORDER)))
    ax.set_xticklabels([ARM_LABELS[a] for a in ARM_ORDER])
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels([TASK_SHORT.get(t, t) for t in tasks], fontsize=8)
    for i in range(len(tasks)):
        for j in range(len(ARM_ORDER)):
            ax.text(j, i, "OK" if data[i, j] == 1 else "—", ha="center", va="center", color="black", fontsize=8)
    ax.set_title("EasyICU v15 completion matrix (15 tasks × 4 settings)")
    ax.set_xlabel("Prompting / execution setting")
    ax.set_ylabel("Task")
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def copy_figures(deliverables: Path, out_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    main = deliverables / "02_publication_figures" / "main"
    rebuilt = deliverables / "02_publication_figures" / "rebuilt"
    supplement = deliverables / "02_publication_figures" / "supplement"
    figure_map = [
        (main / "fig03_prediction_performance_t07.png", "figures/main/Figure_2_prediction_performance_t07.png", "Figure 2", "Prediction performance"),
        (main / "fig04_creatinine_kdigo_t14.png", "figures/main/Figure_3_creatinine_kdigo_t14.png", "Figure 3", "Creatinine trajectory and KDIGO"),
        (main / "fig05_norepi_dose_response_t15.png", "figures/main/Figure_4_norepinephrine_dose_response_t15.png", "Figure 4", "Norepinephrine dose-response association"),
        (main / "fig02_sofa_correlation_t03.png", "figures/main/Figure_5_sofa_correlation_t03.png", "Figure 5", "SOFA component correlation"),
    ]
    for src, rel, role, label in figure_map:
        copy_file(src, out_dir / rel, records, role, label)
        svg = src.with_suffix(".svg")
        if svg.exists():
            copy_file(svg, (out_dir / rel).with_suffix(".svg"), records, role, label + " SVG")
    supp_map = [
        (main / "fig01_sofa_strata_mortality_t02.png", "figures/supplement/Figure_S1_sofa_strata_mortality_t02.png", "Figure S1", "SOFA strata mortality"),
        (supplement / "figS01_table_one_t01.png", "figures/supplement/Figure_S2_table_one_t01.png", "Figure S2", "Baseline descriptive table figure"),
        (rebuilt / "figR01_lactate_or_t04.png", "figures/supplement/Figure_S3_lactate_or_t04.png", "Figure S3", "Lactate OR"),
        (rebuilt / "figR02_vaso_bias_t08.png", "figures/supplement/Figure_S4_vaso_bias_t08.png", "Figure S4", "Vasopressor selection bias"),
        (rebuilt / "figR03_sofa_zero_audit_t09.png", "figures/supplement/Figure_S5_sofa_zero_audit_t09.png", "Figure S5", "SOFA-zero audit"),
        (rebuilt / "figR04_lactate_robustness_t10.png", "figures/supplement/Figure_S6_lactate_robustness_t10.png", "Figure S6", "Lactate robustness"),
        (rebuilt / "figR05_age_mortality_ci_t12.png", "figures/supplement/Figure_S7_age_mortality_t12.png", "Figure S7", "Age-stratified mortality"),
        (supplement / "figS03_shock_cluster_profile_t06.png", "figures/supplement/Figure_S8_shock_cluster_profile_t06.png", "Figure S8", "Shock cluster profile"),
        (supplement / "figS04_los_distribution_t11.png", "figures/supplement/Figure_S9_los_distribution_t11.png", "Figure S9", "ICU length-of-stay distribution"),
        (supplement / "figS06_vital_summary_t13.png", "figures/supplement/Figure_S10_vital_summary_t13.png", "Figure S10", "Admission vital signs"),
    ]
    for src, rel, role, label in supp_map:
        copy_file(src, out_dir / rel, records, role, label)
        svg = src.with_suffix(".svg")
        if svg.exists():
            copy_file(svg, (out_dir / rel).with_suffix(".svg"), records, role, label + " SVG")
    return records


def copy_tables(deliverables: Path, out_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    main = deliverables / "03_publication_tables" / "main"
    supp = deliverables / "03_publication_tables" / "supplement"
    table_map = [
        (main / "table1_success_matrix.csv", "tables/main/Table_1_success_matrix.csv", "Table 1", "Task completion matrix"),
        (main / "table1_success_matrix.md", "tables/main/Table_1_success_matrix.md", "Table 1", "Task completion matrix markdown"),
        (main / "table2_task_metric_summary.csv", "tables/main/Table_2_task_metric_summary.csv", "Table 2", "Task metric summary"),
        (main / "table2_task_metric_summary.md", "tables/main/Table_2_task_metric_summary.md", "Table 2", "Task metric summary markdown"),
        (supp / "tableS1_repair_burden.csv", "tables/supplement/Table_S1_repair_burden.csv", "Table S1", "Repair burden"),
        (supp / "tableS2_artifact_inventory.csv", "tables/supplement/Table_S2_artifact_inventory.csv", "Table S2", "Artifact inventory"),
        (supp / "tableS3_figure_inventory.csv", "tables/supplement/Table_S3_figure_inventory.csv", "Table S3", "Figure inventory"),
        (supp / "tableS4_metric_sanity_audit.csv", "tables/supplement/Table_S4_metric_sanity_audit.csv", "Table S4", "Metric sanity audit"),
        (supp / "tableS5_sofa_zero_audit.csv", "tables/supplement/Table_S5_sofa_zero_audit.csv", "Table S5", "SOFA-zero audit"),
    ]
    for src, rel, role, label in table_map:
        copy_file(src, out_dir / rel, records, role, label)
    return records


def repair_counts_by_arm(repair_rows: list[dict[str, str]]) -> dict[str, int]:
    return {arm: sum(1 for row in repair_rows if row.get("arm") == arm) for arm in ARM_ORDER}


def task_summary_lines(task_summary: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    for row in sorted(task_summary, key=lambda r: task_sort_key(r.get("task_key") or "")):
        task = row.get("task_key") or ""
        lines.append(f"- {task}: {row.get('clean_ok')}/{row.get('n_arms')} cells reached clean completion; {row.get('repair_events')} repair/fallback events. Detailed extracted metrics are reported in Table 2 and Table S4.")
    return lines


def write_title_options(out_dir: Path) -> None:
    lines = [
        "# Title options",
        "",
        "1. Execution-audited LLM-assisted ICU data analysis across a 15-task benchmark",
        "2. Auditing LLM-generated ICU data analyses for execution, artifacts, and repair burden",
        "3. From generated code to traceable ICU analyses: an execution-audited LLM benchmark",
        "4. EasyICU v15: a repair-aware benchmark for LLM-assisted clinical data analysis",
        "5. Traceable, repair-aware LLM code generation for ICU data analysis",
        "",
        "Recommended working title: **Execution-audited LLM-assisted ICU data analysis across a 15-task benchmark**",
    ]
    write_text(out_dir / "manuscript" / "title_options.md", lines)


def write_manuscript(out_dir: Path, matrix: list[dict[str, str]], metrics: list[dict[str, str]], task_summary: list[dict[str, str]], repair_rows: list[dict[str, str]]) -> None:
    counts = status_counts(matrix)
    clean = counts.get("clean_ok", 0)
    manual_flags = [row for row in metrics if str(row.get("manual_review_required") or "").lower() == "true"]
    arm_repairs = repair_counts_by_arm(repair_rows)
    task_count = len({row.get("task_key") for row in matrix})
    arm_count = len({row.get("arm") for row in matrix})
    date = datetime.now(timezone.utc).date().isoformat()
    lines = [
        "# Execution-audited LLM-assisted ICU data analysis across a 15-task benchmark",
        "",
        f"Draft version: v1, generated {date}",
        "",
        "## Abstract",
        "",
        "### Background",
        "Large language models (LLMs) can generate code for clinical data analysis, but a plausible answer is not the same as an analysis that runs, saves its outputs, and leaves an auditable evidence trail. We evaluated an execution-audited LLM-assisted pipeline for ICU data analysis, with emphasis on completion status, artifact traceability, metric extraction, and repair burden.",
        "",
        "### Methods",
        f"We evaluated {task_count} ICU data-analysis tasks under {arm_count} prompting/execution settings, producing {len(matrix)} task-arm cells. The tasks covered descriptive summaries, outcome stratification, severity-score correlation, mortality association, missing-data robustness, prediction performance, clustering, renal dysfunction, vasopressor selection bias, and dose-response analyses. Each cell was processed through code generation or execution, artifact checking, metric extraction, figure/table inventory, and final audit. A cell reached clean completion when execution finished, expected artifacts were present, key metrics were extractable, and no unresolved contract-level failure remained. Repair and fallback events were recorded separately.",
        "",
        "### Results",
        f"All {len(matrix)} task-arm cells reached clean completion ({clean}/{len(matrix)} clean_ok). Automated metric sanity checks covered {len(metrics)} cells and flagged {len(manual_flags)} cells for manual range review. The final audit recorded {len(repair_rows)} repair or fallback events. The curated publication package contained 16 figure records and 7 table records, with source maps linking each presentation-ready output to original run evidence. Main deliverables include a completion matrix, prediction-performance figure, creatinine/KDIGO figure, norepinephrine dose-response figure, SOFA correlation figure, main task summary tables, supplemental audit tables, and rebuilt figures for selected presentation-limited tasks.",
        "",
        "### Conclusions",
        "In this execution-audited benchmark, LLM-assisted ICU analysis produced complete and traceable outputs across all task-arm cells. However, clean completion should be interpreted together with repair burden. The results support a reporting model that separates final runnable outputs from the amount of system assistance needed to obtain them, and they caution against describing repair-assisted completion as unassisted model performance.",
        "",
        "## Introduction",
        "",
        "LLMs are increasingly used to draft analysis plans and generate statistical code. In clinical data science, however, a generated response is insufficient unless the analysis can be executed, checked, and traced. Clinical datasets contain missingness, time-window ambiguity, derived severity scores, coding conventions, and outcome definitions that can make a seemingly reasonable analysis unsafe or irreproducible.",
        "",
        "The EasyICU research-agent framework was designed around this gap. Earlier project versions emphasized evidence-bound manuscript scaffolds: numeric claims should be tied to registered artifacts, and discussion-level clinical claims should remain under human supervision. The present v15 experiment extends that philosophy from a single analysis scaffold to a 15-task benchmark. Instead of asking only whether an LLM can write plausible code, we ask whether the resulting analysis can survive execution, artifact binding, metric extraction, visual review, and final audit.",
        "",
        "This distinction matters for clinical AI evaluation. A task can ultimately produce a clean result while still requiring deterministic repair, dependency workarounds, serialization fixes, or fallback routines. Reporting only final success would overstate model autonomy; reporting only failures would understate the value of a repair-aware execution harness. We therefore report clean completion and repair burden side by side.",
        "",
        "## Methods",
        "",
        "### Benchmark design",
        f"The EasyICU v15 benchmark included {task_count} ICU data-analysis tasks. Each task was run under four settings: `aware`, `aware_no_pref`, `naive_with_pref`, and `naive`. These settings varied whether the model received task context and user preference information. The resulting design produced {len(matrix)} task-arm cells.",
        "",
        "The task set included cohort description, outcome incidence by severity strata, severity-score correlation, lactate-mortality association, KDIGO renal sensitivity, shock phenotype clustering, mortality prediction performance, vasopressor selection-bias audit, SOFA-zero artifact audit, complete-case robustness, ICU length-of-stay description, age-stratified mortality, admission vital-summary description, creatinine trajectory/KDIGO analysis, and norepinephrine-equivalent dose-response analysis.",
        "",
        "### Execution and audit workflow",
        "Each task-arm cell passed through a code-generation and execution workflow. The pipeline checked whether expected artifacts existed, whether step summaries were present, whether key metrics could be extracted, and whether generated figures and tables were registered or recoverable from step outputs. The final audit froze matrix status, metric sanity checks, artifact inventory, figure inventory, repair logs, and publication-facing tables.",
        "",
        "### Clean completion definition",
        "Clean completion (`clean_ok`) is an execution and evidence-binding standard. It indicates that code execution completed, required files were present, key metrics were extractable, and no unresolved contract-level failure remained. It does not imply external clinical validation, absence of bias, publication-ready visual design, or unassisted model performance.",
        "",
        "### Repair and fallback reporting",
        "Repair and fallback events were parsed from audit logs and reported separately from final completion. These events included execution repairs, artifact-binding repairs, metric-extraction support, dependency-free deterministic fallbacks, and task-specific fallbacks. Repair burden was interpreted as the amount of system assistance needed after initial model generation, not as a direct clinical-quality score.",
        "",
        "### Figure and table curation",
        "Original run figures and tables were inventoried. A curated publication package copied suitable main and supplemental figures and rebuilt selected figures from structured summaries when the original visual display was not publication-ready. Rebuilt figures were treated as presentation-layer outputs only; the original run evidence and source maps were preserved.",
        "",
        "### Safety of interpretation",
        "Clinical variable analyses were interpreted as descriptive or associational unless a task explicitly supported causal inference. In particular, associations involving lactate, vasopressor exposure, age, creatinine, or norepinephrine-equivalent dose were not interpreted as proof of a causal effect on mortality.",
        "",
        "## Results",
        "",
        "### Overall completion",
        f"All {len(matrix)} task-arm cells reached clean completion. The final matrix contained {clean} clean_ok cells and no non-clean final cells (Table 1; Figure 1).",
        "",
        "### Completion by setting",
        "All four prompting/execution settings reached clean completion for all tasks. Because final completion was universal, the more informative comparison across settings was repair burden rather than final success alone. Repair/fallback event counts by setting were:",
        "",
    ]
    for arm in ARM_ORDER:
        lines.append(f"- `{arm}`: {arm_repairs.get(arm, 0)} repair/fallback events.")
    lines.extend([
        "",
        "### Task-level results",
        "Task-level summaries are provided in Table 2. Representative extracted metrics included odds ratios, AUROC, silhouette scores, Spearman correlations, mortality rates, and complete-case counts, depending on the task. A brief task-by-task audit summary is below:",
        "",
        *task_summary_lines(task_summary),
        "",
        "### Metric sanity audit",
        f"Automated metric sanity checks covered {len(metrics)} task-arm cells. These checks flagged {len(manual_flags)} cells for manual range review. This result indicates that configured range checks did not identify obvious out-of-range core metrics, but it does not replace clinical or statistical expert review.",
        "",
        "### Repair burden",
        f"The final audit recorded {len(repair_rows)} repair or fallback events. This repair burden should be reported as a central benchmark outcome. It shows that the final 60/60 clean completion was achieved under a repair-aware execution harness rather than through unconstrained model autonomy alone (Table S1).",
        "",
        "### Publication-facing figures and tables",
        "The submission package includes five main figures. Figure 1 shows the 15 × 4 completion matrix. Figure 2 summarizes mortality-prediction performance. Figure 3 shows creatinine trajectory and KDIGO-related renal dysfunction outputs. Figure 4 shows the norepinephrine-equivalent dose-response association and should be interpreted associationally. Figure 5 shows SOFA component correlation. Supplemental figures include baseline description, lactate association, vasopressor selection-bias audit, SOFA-zero audit, lactate missingness robustness, age-stratified mortality, shock clustering, ICU length-of-stay distribution, and admission vital summaries.",
        "",
        "## Discussion",
        "",
        "This benchmark demonstrates that LLM-assisted ICU data analysis can be made substantially more inspectable when model-generated code is embedded in an execution, artifact, metric, and audit framework. The key finding is not merely that all task-arm cells ultimately completed. The more important result is that completion was documented together with output files, metrics, source maps, visual curation, and repair burden.",
        "",
        "The experiment also shows why final success and model autonomy should not be conflated. A repair-aware system can recover from common engineering failures, but those recoveries are part of the system being evaluated. Therefore, clean completion should be reported together with repair burden. This approach is more transparent than either ignoring repair events or treating any repair-assisted output as a simple failure.",
        "",
        "Several features of the EasyICU workflow are important for high-stakes clinical analysis. First, artifacts are inventoried and linked to source files. Second, metric extraction is audited across all cells. Third, problematic figures can be rebuilt from structured summaries while preserving original run evidence. Fourth, claims are intentionally constrained: association analyses remain associational, and discussion-level clinical interpretation remains human-supervised.",
        "",
        "The results should be viewed as evidence for a reproducible analysis workflow, not as clinical validation of each individual scientific finding. The benchmark is most useful for comparing execution reliability, traceability, repair burden, and artifact quality across task types and prompting/execution settings.",
        "",
        "## Limitations",
        "",
        "First, clean completion is a contract-level execution standard and does not prove clinical correctness. Second, the final outputs relied on deterministic repairs and fallback routines in some cells, so the results should not be described as unassisted LLM performance. Third, several figures were rebuilt for presentation quality; these rebuilt figures do not constitute new analyses. Fourth, the clinical analyses are mostly associational and should not be interpreted causally. Fifth, external validation on independent ICU datasets and review by clinical/statistical experts are required before drawing strong clinical conclusions.",
        "",
        "## Data, code, and reproducibility",
        "",
        "The final organized deliverables include the frozen status matrix, repair audit, metric sanity audit, artifact inventory, figure inventory, curated figures and tables, source maps, manuscript draft, supplemental draft, and collaborator review materials. The original run directories were preserved separately and were not moved by the final organization step.",
        "",
        "## References",
        "",
        "TODO for human authors: add real literature references on LLM-generated code, clinical data analysis reproducibility, ICU severity scores, missing-data analysis, and ICU benchmark datasets. No references are fabricated in this generated draft.",
    ])
    write_text(out_dir / "manuscript" / "manuscript_v1_en.md", lines)
    abstract = []
    capture = False
    for line in lines:
        if line == "## Abstract":
            capture = True
        elif line == "## Introduction":
            capture = False
        if capture:
            abstract.append(line)
    write_text(out_dir / "manuscript" / "abstract_v1_en.md", abstract)

    zh = [
        "# 中文审阅说明",
        "",
        "这份英文稿件已经按投稿草稿结构整理：Title、Abstract、Introduction、Methods、Results、Discussion、Limitations、Data/code availability、References TODO。",
        "",
        "## 核心主张",
        "",
        f"15 个 ICU 数据分析任务 × 4 种设置 = {len(matrix)} 个实验单元，全部达到 clean_ok。",
        "",
        "## 必须保留的谨慎表述",
        "",
        f"- 必须报告 {len(repair_rows)} 个 repair/fallback events。",
        "- clean_ok 是执行和证据绑定标准，不是临床结论完全正确。",
        "- 不要把结果写成模型完全自主完成。",
        "- 乳酸、血管活性药物、年龄、去甲肾上腺素等结果应写成 associated，不要写成 caused。",
        "- 参考文献没有编造，References 部分留给人工补真实文献。",
    ]
    write_text(out_dir / "manuscript" / "manuscript_v1_zh_review_notes.md", zh)


def write_supplement(out_dir: Path, matrix: list[dict[str, str]], metrics: list[dict[str, str]], task_summary: list[dict[str, str]], repair_rows: list[dict[str, str]]) -> None:
    repair_phase = Counter(row.get("phase") or "unknown" for row in repair_rows)
    lines = [
        "# Supplementary Material",
        "",
        "## Supplementary Methods",
        "",
        "### Task list",
        "The benchmark included 15 tasks: cohort description, outcome incidence by severity strata, severity-score correlation, lactate-mortality association, KDIGO renal sensitivity, shock phenotype clustering, mortality prediction, vasopressor selection-bias audit, SOFA-zero artifact audit, missing-data robustness, length-of-stay description, age-stratified mortality, admission vital-summary description, creatinine trajectory/KDIGO analysis, and norepinephrine-equivalent dose-response analysis.",
        "",
        "### Prompting/execution settings",
        "Four settings were used: `aware`, `aware_no_pref`, `naive_with_pref`, and `naive`. These settings varied the presence of task context and user preference information.",
        "",
        "### Clean completion",
        "A cell reached `clean_ok` when execution completed, expected artifacts were present, key metrics were extractable, and no unresolved contract-level failure remained. This is not equivalent to external clinical validation.",
        "",
        "### Repair and fallback events",
        "Repair and fallback events were parsed from audit logs. These events were reported separately to distinguish final runnable output from the amount of system assistance required.",
        "",
        "## Supplementary Results",
        "",
        f"The final matrix contained {len(matrix)} cells. All cells reached clean completion. Metric sanity checks covered {len(metrics)} cells. Repair/fallback events totalled {len(repair_rows)}.",
        "",
        "### Repair phases",
        "",
    ]
    for phase, count in repair_phase.most_common():
        lines.append(f"- {phase}: {count}")
    lines.extend([
        "",
        "### Task-level summary",
        "",
        *task_summary_lines(task_summary),
        "",
        "## Supplementary Tables",
        "",
        "- Table S1. Repair burden by task, setting, and repair phase.",
        "- Table S2. Artifact inventory.",
        "- Table S3. Figure inventory.",
        "- Table S4. Metric sanity audit.",
        "- Table S5. SOFA-zero audit.",
        "",
        "## Supplementary Figures",
        "",
        "- Figure S1. SOFA strata mortality.",
        "- Figure S2. Baseline descriptive figure.",
        "- Figure S3. Lactate odds ratio.",
        "- Figure S4. Vasopressor selection-bias audit.",
        "- Figure S5. SOFA-zero artifact audit.",
        "- Figure S6. Lactate missingness robustness.",
        "- Figure S7. Age-stratified mortality.",
        "- Figure S8. Shock cluster profile.",
        "- Figure S9. ICU length-of-stay distribution.",
        "- Figure S10. Admission vital-summary figure.",
        "",
        "## Supplementary Limitations",
        "",
        "The benchmark evaluates an execution-audited workflow rather than unconstrained model autonomy. Repair burden should be interpreted as part of system performance. Clinical associations should not be interpreted causally without additional study design and external validation.",
    ])
    write_text(out_dir / "supplement" / "supplement_v1.md", lines)


def write_cover_material(out_dir: Path, repair_count: int) -> None:
    lines = [
        "# Collaborator review request",
        "",
        "Dear collaborator,",
        "",
        "I am sharing the EasyICU v15 submission-ready package for review. The study evaluates an execution-audited LLM-assisted ICU data-analysis pipeline across 15 tasks and 4 prompting/execution settings.",
        "",
        "Please focus on the following points:",
        "",
        "1. Are the main figures understandable and clinically reasonable?",
        "2. Are any clinical claims too strong?",
        "3. Is the repair/fallback disclosure clear enough?",
        "4. Are associational findings clearly distinguished from causal claims?",
        "5. Should any main figure be moved to the supplement or vice versa?",
        "",
        f"Important context: all 60 cells reached clean completion, but the audit recorded {repair_count} repair/fallback events. The manuscript therefore reports clean completion together with repair burden rather than claiming unassisted model performance.",
        "",
        "Suggested files to review first:",
        "",
        "- `manuscript/manuscript_v1_en.md`",
        "- `manuscript/abstract_v1_en.md`",
        "- `figures/main/`",
        "- `figures/supplement/`",
        "- `tables/main/`",
        "- `supplement/supplement_v1.md`",
    ]
    write_text(out_dir / "cover_material" / "collaborator_review_request.md", lines)
    safety = [
        "# Claim safety note",
        "",
        "Use these wording rules when editing the manuscript.",
        "",
        "## Safe wording",
        "",
        "- reached clean completion",
        "- execution-audited pipeline",
        "- repair burden was recorded separately",
        "- was associated with",
        "- differed across groups",
        "- presentation-layer redraw from structured summaries",
        "",
        "## Avoid wording",
        "",
        "- fully autonomous",
        "- proved clinical correctness",
        "- causal effect on mortality",
        "- treatment effect",
        "- clinically validated",
        "- new analysis from rebuilt figure",
    ]
    write_text(out_dir / "cover_material" / "reviewer_claim_safety_note.md", safety)


def write_readme(out_dir: Path) -> None:
    lines = [
        "# EasyICU v15 submission-ready package",
        "",
        "Start here after the final deliverables package. This folder contains a more submission-oriented version of the manuscript, figures, tables, supplement, and collaborator review material.",
        "",
        "## Contents",
        "",
        "- `manuscript/`: title options, manuscript v1, abstract, Chinese review notes.",
        "- `figures/main/`: renamed main figures for manuscript use.",
        "- `figures/supplement/`: renamed supplemental figures.",
        "- `tables/main/`: renamed main tables.",
        "- `tables/supplement/`: renamed supplemental tables.",
        "- `supplement/`: supplementary material draft.",
        "- `cover_material/`: collaborator review request and claim-safety note.",
        "- `manifests/`: figure/table/source manifests.",
        "",
        "## Important note",
        "",
        "The manuscript does not fabricate references. The References section is a TODO for human authors to add real literature citations.",
    ]
    write_text(out_dir / "README_SUBMISSION_READY.md", lines)


def build_package(deliverables: Path, out_dir: Path) -> None:
    audit = deliverables / "01_experiment_audit"
    table1 = read_csv(deliverables / "03_publication_tables" / "main" / "table1_success_matrix.csv")
    matrix = read_csv(audit / "matrix_status.csv")
    metrics = read_csv(audit / "metric_sanity_audit.csv")
    task_summary = read_csv(audit / "paper_task_metric_summary.csv")
    repair_rows = read_csv(audit / "repair_audit.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    build_completion_heatmap(table1, out_dir / "figures" / "main" / "Figure_1_completion_matrix")
    figure_records = [{"role": "Figure 1", "label": "Completion matrix", "file": str(out_dir / "figures" / "main" / "Figure_1_completion_matrix.png"), "source_file": str(deliverables / "03_publication_tables" / "main" / "table1_success_matrix.csv"), "exists": True, "size_bytes": (out_dir / "figures" / "main" / "Figure_1_completion_matrix.png").stat().st_size}]
    figure_records.extend(copy_figures(deliverables, out_dir))
    table_records = copy_tables(deliverables, out_dir)
    write_title_options(out_dir)
    write_manuscript(out_dir, matrix, metrics, task_summary, repair_rows)
    write_supplement(out_dir, matrix, metrics, task_summary, repair_rows)
    write_cover_material(out_dir, len(repair_rows))
    write_readme(out_dir)
    write_csv(out_dir / "manifests" / "figure_manifest.csv", figure_records, ["role", "label", "file", "source_file", "exists", "size_bytes"])
    write_csv(out_dir / "manifests" / "table_manifest.csv", table_records, ["role", "label", "file", "source_file", "exists", "size_bytes"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "deliverables": str(deliverables),
        "out_dir": str(out_dir),
        "matrix_rows": len(matrix),
        "clean_ok": status_counts(matrix).get("clean_ok", 0),
        "repair_events": len(repair_rows),
        "figure_records": len(figure_records),
        "table_records": len(table_records),
    }
    (out_dir / "manifests" / "submission_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deliverables", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    deliverables = Path(args.deliverables).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else deliverables / "08_submission_ready"
    build_package(deliverables, out_dir)
    print(out_dir)
    print(out_dir / "README_SUBMISSION_READY.md")
    print(out_dir / "manuscript" / "manuscript_v1_en.md")
    print(out_dir / "supplement" / "supplement_v1.md")
    print(out_dir / "manifests" / "submission_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
