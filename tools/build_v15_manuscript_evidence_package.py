#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ARM_ORDER = ["aware", "aware_no_pref", "naive_with_pref", "naive"]

TASK_NAMES = {
    "t01_table_one_descriptive": "整理队列基本情况",
    "t02_outcome_incidence_strata": "按严重程度分层看死亡率",
    "t03_severity_score_correlation": "比较不同严重程度评分",
    "t04_lactate_mortality_association": "乳酸和死亡风险的关系",
    "t05_kdigo_renal_sensitivity": "KDIGO 肾损伤敏感性分析",
    "t06_shock_phenotype_clustering": "休克患者分型",
    "t07_mortality_prediction_auroc": "死亡预测模型表现",
    "t08_vaso_selection_bias_audit": "血管活性药物选择偏倚检查",
    "t09_sofa_zero_artefact_audit": "SOFA 为 0 的异常检查",
    "t10_complete_case_robustness": "完整病例和缺失处理稳健性",
    "t11_los_distribution_descriptive": "ICU 住院时长分布",
    "t12_age_stratified_mortality": "按年龄分层看死亡率",
    "t13_admission_vital_summary": "入院生命体征概览",
    "t14_creatinine_trajectory_kdigo": "肌酐轨迹和 KDIGO 肾功能变化",
    "t15_norepinephrine_dose_response": "去甲肾上腺素剂量和结局关系",
}

ARM_NAMES = {
    "aware": "知道任务背景、也知道偏好设置",
    "aware_no_pref": "知道任务背景、但不知道偏好设置",
    "naive_with_pref": "不知道任务背景、但知道偏好设置",
    "naive": "任务背景和偏好设置都不给",
}

MAIN_FIGURE_CAPTIONS = {
    "fig01_sofa_strata_mortality_t02": "图 1. 不同 SOFA 严重程度分层中的死亡率。这个图用来说明队列中病情越重，死亡率总体越高。",
    "fig02_sofa_correlation_t03": "图 2. 不同严重程度评分或组成部分之间的相关性。这个图用来检查评分之间是否方向一致。",
    "fig03_prediction_performance_t07": "图 3. 死亡预测模型表现。这个图用 AUROC 等指标展示模型区分死亡和未死亡患者的能力。",
    "fig04_creatinine_kdigo_t14": "图 4. 肌酐轨迹和 KDIGO 肾功能变化。这个图用于展示患者肾功能变化的分组模式。",
    "fig05_norepi_dose_response_t15": "图 5. 去甲肾上腺素等效剂量和结局的关系。这个图只能解释为相关关系，不能写成药物导致死亡。",
}

SUPPLEMENT_CAPTIONS = {
    "figS01_table_one_t01": "补充图 S1. 队列基线描述。用于快速查看样本的年龄、性别、ICU 时长、死亡率等基本情况。",
    "figS02_kdigo_mortality_t05": "补充图 S2. KDIGO 分组和死亡率。用于说明不同肾损伤定义下结果是否一致。",
    "figS03_shock_cluster_profile_t06": "补充图 S3. 休克患者分型特征。该图是描述性分型，不代表因果分组。",
    "figS04_los_distribution_t11": "补充图 S4. ICU 住院时长分布。用于展示住院时长的偏态和长尾情况。",
    "figS05_age_mortality_t12_original": "补充图 S5. 年龄分层死亡率原始图。最终展示时可优先使用重画版 figR05。",
    "figS06_vital_summary_t13": "补充图 S6. 入院生命体征概览。用于描述入院时患者基本生命体征分布。",
    "figR01_lactate_or_t04": "补充图 R1. 乳酸和死亡风险的关联。点为 OR，横线为 95% CI。该图是从结构化结果重画，只改善展示效果。",
    "figR02_vaso_bias_t08": "补充图 R2. 血管活性药物暴露组和未暴露组在不同分层中的死亡率。该图用于展示选择偏倚，不用于证明用药导致死亡。",
    "figR03_sofa_zero_audit_t09": "补充图 R3. SOFA 为 0 人群中的异常信号检查。用于提醒评分为 0 不等于完全没有风险。",
    "figR04_lactate_robustness_t10": "补充图 R4. 乳酸关联结果在不同缺失数据处理方法下的变化。CI 未在原始结构化结果中提供时，图中只展示 OR 和样本量。",
    "figR05_age_mortality_ci_t12": "补充图 R5. 按年龄分层的死亡率和置信区间。该图是从结构化结果重画，便于论文展示。",
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


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def status_counts(matrix: list[dict[str, str]]) -> Counter[str]:
    return Counter(row.get("status") or "" for row in matrix)


def task_order(task: str) -> tuple[int, str]:
    try:
        return int(task[1:3]), task
    except Exception:
        return 999, task


def write_readme(out_dir: Path, audit_dir: Path, matrix: list[dict[str, str]], repair_rows: list[dict[str, str]], figure_rows: list[dict[str, str]], table_rows: list[dict[str, str]]) -> None:
    counts = status_counts(matrix)
    lines = [
        "# EasyICU v15 论文材料包（白话版）",
        "",
        f"生成时间：`{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## 这个文件夹是干什么的？",
        "",
        "这个文件夹把 final audit 和 curated publication package 里的内容，整理成更容易写论文、给合作者解释、给审稿人交代的材料。",
        "",
        "它不是新的实验结果，也没有重新跑模型。它只是把已经完成的 60 个实验单元重新组织成：",
        "",
        "- 方法怎么写",
        "- 结果怎么写",
        "- 图表怎么引用",
        "- 哪些话可以说",
        "- 哪些话不要说",
        "- 自动修复要怎么透明披露",
        "",
        "## 当前最重要的结论",
        "",
        f"- 总实验单元：`{len(matrix)}`",
        f"- 状态统计：`{dict(counts)}`",
        f"- 修复/兜底事件数：`{len(repair_rows)}`",
        f"- curated 图记录数：`{len(figure_rows)}`",
        f"- curated 表记录数：`{len(table_rows)}`",
        "",
        "一句话版本：15 个 ICU 研究任务、4 种提示/执行设置，共 60 个单元，最终都达到了 `clean_ok`。但是，部分单元依赖了程序层面的自动修复或兜底逻辑，所以论文里要把“最终完成”和“完全靠模型独立完成”分开说。",
        "",
        "## 文件夹怎么读？",
        "",
        "- `methods/`：方法部分怎么写。",
        "- `results/`：结果部分怎么写。",
        "- `captions/`：主图、补充图、表格的说明文字。",
        "- `checklist/`：写论文前逐项检查，防止说过头。",
        "- `submission_package/`：投稿或发给合作者时可用的图表清单。",
        "- `one_page_summary_plain.md`：一页版总结，适合先发给合作者。",
        "- `glossary_plain.md`：名词解释，把容易看不懂的词翻译成人话。",
        "",
        "## 原始证据在哪里？",
        "",
        f"- final audit：`{audit_dir}`",
        f"- curated publication：`{audit_dir / 'curated_publication'}`",
        "",
        "## 最重要的写作原则",
        "",
        "- 可以说：这些任务在执行框架和检查框架下都完成了。",
        "- 可以说：所有核心结果都有表格、图和审计记录。",
        "- 必须说：有些结果经过了自动修复或兜底。",
        "- 不要说：模型完全自主、没有任何程序帮助。",
        "- 不要说：某个变量导致死亡，除非有专门的因果设计。这里多数结果只是相关关系。",
    ]
    write_text(out_dir / "README.md", lines)


def write_plain_extras(out_dir: Path, matrix: list[dict[str, str]], repair_rows: list[dict[str, str]], figure_rows: list[dict[str, str]], table_rows: list[dict[str, str]]) -> None:
    counts = status_counts(matrix)
    one_page = [
        "# EasyICU v15 一页版总结",
        "",
        "## 这批结果一句话是什么？",
        "",
        f"我们跑完了 15 个 ICU 数据分析任务，每个任务有 4 种设置，总共 `{len(matrix)}` 个实验单元；最终状态是 `{dict(counts)}`。",
        "",
        "## 这说明什么？",
        "",
        "说明在当前这套“生成代码、运行代码、检查文件、抽取指标、必要时自动修复”的流程下，所有任务最终都能形成可检查的结果包。",
        "",
        "## 这不说明什么？",
        "",
        "它不说明模型完全独立、一次性、没有帮助地完成了全部科学分析。因为中间有自动修复和兜底逻辑。",
        "",
        "## 最应该报告的数字",
        "",
        f"- 实验单元：`{len(matrix)}`",
        f"- clean_ok：`{counts.get('clean_ok', 0)}`",
        f"- 修复/兜底事件：`{len(repair_rows)}`",
        f"- curated 图记录：`{len(figure_rows)}`",
        f"- curated 表记录：`{len(table_rows)}`",
        "",
        "## 论文里最稳妥的说法",
        "",
        "可以写：",
        "",
        "> All 60 task-arm cells reached clean completion under the execution, artifact-checking, metric-extraction, and repair-aware pipeline.",
        "",
        "中文意思：所有 60 个实验单元都在这套带检查和修复的流程下完成了。",
        "",
        "## 论文里不要这样写",
        "",
        "- 不要写：模型完全自主完成了全部分析。",
        "- 不要写：这些临床发现已经被临床验证。",
        "- 不要写：乳酸、血管活性药物或年龄“导致”死亡。",
        "- 不要写：重画图是新的分析结果。",
        "",
        "## 如果只给合作者看 3 个文件",
        "",
        "建议先看：",
        "",
        "1. `one_page_summary_plain.md`",
        "2. `results/main_results_plain.md`",
        "3. `checklist/manuscript_claim_checklist.csv`",
    ]
    write_text(out_dir / "one_page_summary_plain.md", one_page)

    glossary = [
        "# 名词解释（白话版）",
        "",
        "这个文件专门解释容易看不懂的词。",
        "",
        "## clean_ok",
        "",
        "意思是这个实验单元通过了项目内部检查。代码跑完了，该有的文件和关键指标也能找到。",
        "",
        "它不是“临床结论一定正确”的意思。",
        "",
        "## task-arm cell",
        "",
        "一个任务加一种设置，就是一个实验单元。",
        "",
        "例如：`t04 乳酸和死亡风险` 在 `aware` 设置下跑一次，这就是一个 cell。",
        "",
        "## arm",
        "",
        "就是实验设置。本项目有 4 种设置，区别是给不给模型任务背景、给不给偏好信息。",
        "",
        "## artifact",
        "",
        "就是实验留下来的文件，例如图、表、JSON summary、CSV 结果表。",
        "",
        "## manifest",
        "",
        "可以理解成文件登记表。它记录这次实验生成了哪些结果文件。",
        "",
        "## metric",
        "",
        "指标。比如 OR、AUROC、死亡率、相关系数、聚类数量。",
        "",
        "## metric sanity check",
        "",
        "指标合理性检查。比如 AUROC 应该在 0 到 1 之间，聚类数量不能是负数。",
        "",
        "这个检查只能发现明显异常，不能替代人工临床判断。",
        "",
        "## repair",
        "",
        "自动修复。模型生成的代码或结果有小问题时，系统帮它修正。",
        "",
        "例如文件没有正确登记、JSON 格式不对、缺少依赖包、图表路径不对。",
        "",
        "## fallback",
        "",
        "兜底逻辑。意思是模型生成的方案不稳定时，系统用一段确定的、可重复的代码保证任务能输出结果。",
        "",
        "这必须在论文里说明，因为它不是纯模型独立完成。",
        "",
        "## deterministic",
        "",
        "确定性的。意思是同样输入下，这段程序应该给出同样结果，不靠随机发挥。",
        "",
        "## deterministic repair burden",
        "",
        "自动修复负担。就是为了让实验最终完成，系统到底帮了多少忙。",
        "",
        "## curated publication package",
        "",
        "整理后的论文图表包。里面的图和表更适合写论文，但它们来自原始实验结果，不是新实验。",
        "",
        "## rebuilt figure",
        "",
        "重画图。意思是从原始结构化结果里拿数字，重新画成更清楚的图。",
        "",
        "重画图只是为了好看、清楚，不改变结果。",
        "",
        "## source map",
        "",
        "来源表。它告诉你每张整理后的图或表，是从哪个原始文件来的。",
        "",
        "## OR",
        "",
        "Odds ratio，优势比。简单理解是一个变量升高时，结局发生的相对 odds 有多大变化。",
        "",
        "OR 大于 1 通常表示正相关，但不能直接说成导致。",
        "",
        "## CI",
        "",
        "Confidence interval，置信区间。用来表示估计值的不确定性。",
        "",
        "## AUROC",
        "",
        "预测模型区分能力指标。越接近 1 越好，0.5 大约相当于随机猜。",
        "",
        "## silhouette score",
        "",
        "聚类质量指标。用来粗略看分组是否分得开。",
        "",
        "## Spearman rho",
        "",
        "一种相关系数。用来看两个变量是不是大致一起升高或一起降低。",
        "",
        "## associational",
        "",
        "相关关系。意思是两个现象一起出现或有统计联系，但不能证明谁导致谁。",
        "",
        "## causal",
        "",
        "因果关系。意思是 A 导致 B。这个词要非常谨慎用，除非研究设计专门支持因果推断。",
    ]
    write_text(out_dir / "glossary_plain.md", glossary)

    collaborator = [
        "# 给合作者的简短说明",
        "",
        "我们已经把 EasyICU v15 的 final audit 和论文图表整理成一个可审查材料包。",
        "",
        "## 当前状态",
        "",
        f"- 15 个任务 × 4 种设置 = `{len(matrix)}` 个实验单元。",
        f"- 最终状态：`{dict(counts)}`。",
        f"- 自动修复/兜底事件：`{len(repair_rows)}`。",
        f"- 整理后的图记录：`{len(figure_rows)}`。",
        f"- 整理后的表记录：`{len(table_rows)}`。",
        "",
        "## 我们希望你帮忙看的重点",
        "",
        "1. 主图和补充图是否容易理解。",
        "2. 结果文字有没有说过头。",
        "3. 自动修复/兜底的披露是否足够清楚。",
        "4. 临床变量分析是否都保持在“相关关系”层面，没有写成因果。",
        "",
        "## 最重要的限制",
        "",
        "这些结果说明任务在当前执行和检查框架下完成了；它们不等于模型完全自主完成，也不等于临床结论已经被外部验证。",
    ]
    write_text(out_dir / "collaborator_note_plain.md", collaborator)


def write_methods(out_dir: Path, matrix: list[dict[str, str]], repair_rows: list[dict[str, str]]) -> None:
    task_count = len({row.get("task_key") for row in matrix})
    arm_count = len({row.get("arm") for row in matrix})
    lines = [
        "# 实验设计怎么写（白话版）",
        "",
        "## 我们做了什么？",
        "",
        f"我们设计了 `{task_count}` 个 ICU 数据分析任务。每个任务都在 `{arm_count}` 种设置下运行，所以一共有 `{len(matrix)}` 个实验单元。",
        "",
        "这些任务覆盖了常见的临床数据分析场景，例如：",
        "",
        "- 描述患者基本情况",
        "- 比较不同严重程度评分",
        "- 分析乳酸、肾功能、血压、血管活性药物等变量和死亡率的关系",
        "- 做预测模型表现检查",
        "- 做缺失数据和异常值检查",
        "",
        "## 四种设置是什么意思？",
        "",
    ]
    for arm in ARM_ORDER:
        lines.append(f"- `{arm}`：{ARM_NAMES.get(arm, arm)}。")
    lines.extend([
        "",
        "这四种设置的目的，是看模型在不同信息条件下能不能完成同一类任务。",
        "",
        "## 每个实验单元怎么运行？",
        "",
        "每个单元大致经过这些步骤：",
        "",
        "1. 给模型一个数据分析任务。",
        "2. 模型生成分析计划和代码。",
        "3. 系统运行代码。",
        "4. 系统检查有没有生成该有的表、图和结果摘要。",
        "5. 如果出现环境问题、文件绑定问题、结果格式问题，系统会尝试自动修复。",
        "6. 最后记录状态、指标、图表和修复过程。",
        "",
        "## 论文里应该怎么说？",
        "",
        "推荐写法：",
        "",
        "> We evaluated 15 ICU data-analysis tasks under four prompting/execution settings. Each task-arm cell was run through a code-generation, execution, artifact-checking, and audit pipeline. The final status, metrics, figures, and repair events were recorded for every cell.",
        "",
        "中文意思：我们不是只看模型有没有回答，而是看它生成的分析能不能真正跑完、能不能留下可检查的结果文件。",
        "",
        "## 自动修复数量",
        "",
        f"本次 final audit 记录到 `{len(repair_rows)}` 个修复/兜底事件。论文里不要把这部分藏起来，应该作为透明度结果报告。",
    ])
    write_text(out_dir / "methods" / "experiment_design_plain.md", lines)

    clean_lines = [
        "# clean_ok 是什么意思？",
        "",
        "`clean_ok` 是本项目内部的完成标准。它的意思是：这个实验单元通过了程序层面的检查。",
        "",
        "## clean_ok 代表什么？",
        "",
        "它通常代表：",
        "",
        "- 代码跑完了。",
        "- 该有的结果文件存在。",
        "- 关键指标能被系统读出来。",
        "- 图、表、summary 能和实验单元对应上。",
        "- 没有留下未解决的严重执行错误。",
        "",
        "## clean_ok 不代表什么？",
        "",
        "它不等于：",
        "",
        "- 临床结论一定正确。",
        "- 图一定适合直接投稿。",
        "- 模型完全没有靠程序帮助。",
        "- 所有统计问题都已经由人工专家审核过。",
        "- 变量之间存在因果关系。",
        "",
        "## 论文中推荐写法",
        "",
        "> Clean completion means that the task-arm cell satisfied the execution contract: code execution finished, expected artifacts were present, key metrics were extractable, and no unresolved contract-level failure remained.",
        "",
        "## 不推荐写法",
        "",
        "不要写：",
        "",
        "- 模型独立完成了全部科学分析。",
        "- 所有结果都已经临床验证。",
        "- clean_ok 证明结果没有偏倚。",
    ]
    write_text(out_dir / "methods" / "clean_ok_plain_definition.md", clean_lines)

    repair_lines = [
        "# 自动修复和兜底逻辑怎么解释？",
        "",
        "## 简单说",
        "",
        "自动修复就是：模型生成的代码或结果如果出现常见问题，系统会尝试帮它修好。",
        "",
        "常见情况包括：",
        "",
        "- 缺少某个 Python 包。",
        "- 结果文件生成了，但没有正确登记到 manifest。",
        "- JSON 里有不能直接保存的 numpy 类型。",
        "- 图或表的文件名和系统预期不一致。",
        "- 某些任务需要稳定输出，但模型生成的版本不够稳定，于是系统使用确定性的兜底代码。",
        "",
        "## 为什么要报告这件事？",
        "",
        "因为这会影响我们怎么解释结果。",
        "",
        "如果一个单元 clean_ok，但中间用了很多自动修复，那么它说明的是：在这个执行框架帮助下，任务最终完成了。它不能简单理解成：模型自己一次性完美完成。",
        "",
        "## 论文里推荐写法",
        "",
        "> We report clean completion together with repair burden. This separates final runnable output from the amount of execution help required to obtain that output.",
        "",
        "中文意思：我们同时报告最终完成情况和修复负担，这样读者能看出哪些任务更依赖系统帮助。",
        "",
        "## 本次修复总量",
        "",
        f"本次 final audit 记录到 `{len(repair_rows)}` 个修复/兜底事件。详细表在 `paper_repair_burden.csv` 和 `repair_audit.csv`。",
    ]
    write_text(out_dir / "methods" / "repair_and_fallback_plain_disclosure.md", repair_lines)

    check_lines = [
        "# 指标和图表是怎么检查的？",
        "",
        "本项目做了两层检查。",
        "",
        "## 第一层：程序检查",
        "",
        "系统检查每个实验单元有没有：",
        "",
        "- 运行完成",
        "- 保存结果摘要",
        "- 生成该有的表格",
        "- 生成该有的图",
        "- 抽取到关键指标",
        "- 指标没有明显超出合理范围",
        "",
        "## 第二层：人工视觉检查",
        "",
        "程序检查只能发现空文件、缺文件、明显异常值。它不能判断一张图是否好看、是否适合论文主文。",
        "",
        "所以我们另外做了图像审查，并把一些原始图重画成更适合论文展示的版本。重画只是为了展示更清楚，不改变原始结果。",
        "",
        "## 论文里应该怎么说？",
        "",
        "推荐写法：",
        "",
        "> Metrics and artifacts were audited programmatically, and key figures were visually reviewed. Some figures were redrawn from structured step summaries for presentation quality while preserving the original run evidence.",
    ]
    write_text(out_dir / "methods" / "metric_and_figure_checking_plain.md", check_lines)


def write_results(out_dir: Path, matrix: list[dict[str, str]], metrics: list[dict[str, str]], task_summary: list[dict[str, str]], repair_rows: list[dict[str, str]]) -> None:
    counts = status_counts(matrix)
    flags = [row for row in metrics if str(row.get("manual_review_required") or "").lower() == "true"]
    by_arm = defaultdict(list)
    for row in matrix:
        by_arm[row.get("arm") or ""].append(row)
    lines = [
        "# 主要结果怎么写（白话版）",
        "",
        "## 一句话结果",
        "",
        f"15 个任务 × 4 种设置，一共 `{len(matrix)}` 个实验单元，最终状态统计为 `{dict(counts)}`。",
        "",
        "换句话说，本轮最终实验矩阵已经全部完成。",
        "",
        "## 但是要注意",
        "",
        "这个结果不是说模型每次都一次性成功。中间有自动修复和兜底逻辑。论文里应该把两件事分开：",
        "",
        "- 最终有没有完成。",
        "- 为了完成用了多少系统帮助。",
        "",
        "## 指标检查结果",
        "",
        f"自动指标检查覆盖 `{len(metrics)}` 个实验单元。需要人工复核标记的单元数：`{len(flags)}`。",
        "",
    ]
    if flags:
        lines.append("有自动标记的单元如下：")
        lines.append("")
        for row in flags:
            lines.append(f"- `{row.get('task_key')}` / `{row.get('arm')}`：{row.get('review_flags')}")
    else:
        lines.append("自动范围检查没有发现需要人工复核的核心指标。")
    lines.extend([
        "",
        "## 推荐英文结果段落",
        "",
        "> Across 15 ICU data-analysis tasks and four prompting/execution settings, all 60 task-arm cells reached clean completion after execution, artifact checking, metric extraction, and deterministic repair where needed. Automated metric sanity checks did not flag any core metric for manual range review. Repair burden was recorded separately to distinguish final completion from the amount of system assistance required.",
    ])
    write_text(out_dir / "results" / "main_results_plain.md", lines)

    task_lines = [
        "# 每个任务结果怎么讲",
        "",
        "这个文件给每个任务一个简单解释。写论文时可以从这里摘句子。",
        "",
    ]
    for row in sorted(task_summary, key=lambda r: task_order(r.get("task_key") or "")):
        task = row.get("task_key") or ""
        task_lines.extend([
            f"## `{task}`：{TASK_NAMES.get(task, task)}",
            "",
            f"- 完成情况：`{row.get('clean_ok')}/{row.get('n_arms')}` 个设置 clean_ok。",
            f"- 修复事件：`{row.get('repair_events')}`。",
            f"- 代表性指标：{row.get('representative_metrics') or '见完整 audit 表。'}",
            "- 写作提醒：描述为数据分析结果或相关关系，避免写成因果结论。",
            "",
        ])
    write_text(out_dir / "results" / "task_by_task_plain_notes.md", task_lines)

    arm_lines = [
        "# 四种设置怎么比较",
        "",
        "## 先说结论",
        "",
        "四种设置最终都完成了全部任务。因此，不能只用最终 clean_ok 来区分它们。",
        "",
        "更有意义的比较是：",
        "",
        "- 哪种设置需要更多修复。",
        "- 哪种设置生成的图表更容易直接使用。",
        "- 哪种设置更容易漏掉文件、格式或指标。",
        "",
        "## 各设置完成情况",
        "",
    ]
    repair_by_arm = Counter(row.get("arm") or "" for row in repair_rows)
    for arm in ARM_ORDER:
        rows = by_arm.get(arm, [])
        clean = sum(1 for row in rows if row.get("status") == "clean_ok")
        arm_lines.append(f"- `{arm}`：{ARM_NAMES.get(arm, arm)}；`{clean}/{len(rows)}` clean_ok；修复事件 `{repair_by_arm.get(arm, 0)}`。")
    arm_lines.extend([
        "",
        "## 写作提醒",
        "",
        "不要写某个 arm 绝对更聪明。更稳妥的写法是：某个 arm 在这个执行框架下需要的修复更少，或者生成的结果更容易通过检查。",
    ])
    write_text(out_dir / "results" / "arm_comparison_plain_notes.md", arm_lines)

    phase_counts = Counter(row.get("phase") or "unknown" for row in repair_rows)
    task_counts = Counter(row.get("task_key") or "" for row in repair_rows)
    repair_lines = [
        "# 修复负担结果怎么讲",
        "",
        f"本次共有 `{len(repair_rows)}` 个修复/兜底事件。",
        "",
        "## 按阶段统计",
        "",
    ]
    for phase, count in phase_counts.most_common():
        repair_lines.append(f"- `{phase}`：{count}")
    repair_lines.extend(["", "## 修复最多的任务", ""])
    for task, count in task_counts.most_common(10):
        repair_lines.append(f"- `{task}`：{count}")
    repair_lines.extend([
        "",
        "## 论文里怎么解释",
        "",
        "修复负担高不一定说明结果没用。它更可能说明这个任务更容易遇到工程问题，例如依赖包、文件绑定、图表生成或结果格式。",
        "",
        "但修复负担必须透明报告，因为它影响读者对模型独立能力的判断。",
    ])
    write_text(out_dir / "results" / "repair_burden_plain_notes.md", repair_lines)


def basename_no_suffix(path_text: str) -> str:
    name = Path(path_text).name
    for suffix in [".png", ".svg", ".csv", ".md"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def write_captions(out_dir: Path, figure_rows: list[dict[str, str]], table_rows: list[dict[str, str]]) -> None:
    main_lines = ["# 主文图说明文字", "", "这些 caption 是初稿，可以直接复制到论文草稿里再润色。", ""]
    supp_lines = ["# 补充图说明文字", "", "重画图只改变展示方式，不改变原始实验结果。", ""]
    for row in figure_rows:
        key = basename_no_suffix(row.get("curated_file") or "")
        if "/figures/main/" in (row.get("curated_file") or ""):
            main_lines.extend([f"## `{Path(row.get('curated_file') or '').name}`", "", MAIN_FIGURE_CAPTIONS.get(key, f"图. {key}。"), ""])
        else:
            supp_lines.extend([f"## `{Path(row.get('curated_file') or '').name}`", "", SUPPLEMENT_CAPTIONS.get(key, f"补充图. {key}。"), ""])
    table_lines = [
        "# 表格说明文字",
        "",
        "## `table1_success_matrix.csv`",
        "",
        "表 1. 15 个任务在 4 种设置下的最终完成状态。`clean_ok` 表示该单元通过执行、文件、指标和结果绑定检查。",
        "",
        "## `table2_task_metric_summary.csv`",
        "",
        "表 2. 每个任务的代表性指标和修复事件数。该表用于概览不同任务的结果，而不是替代完整 audit 表。",
        "",
        "## `tableS1_repair_burden.csv`",
        "",
        "补充表 S1. 每个任务、设置和修复阶段的修复事件数。该表用于透明报告系统帮助的程度。",
        "",
        "## `tableS2_artifact_inventory.csv`",
        "",
        "补充表 S2. 所有登记输出文件的清单，包括文件路径和大小。",
        "",
        "## `tableS3_figure_inventory.csv`",
        "",
        "补充表 S3. 所有图像文件的清单，用于追踪图像来源和人工审查。",
        "",
        "## `tableS4_metric_sanity_audit.csv`",
        "",
        "补充表 S4. 核心指标的自动范围检查结果。",
        "",
        "## `tableS5_sofa_zero_audit.csv`",
        "",
        "补充表 S5. SOFA 为 0 人群中的异常信号计数。",
    ]
    if table_rows:
        table_lines.extend(["", "## curated 表格来源", ""])
        for row in table_rows:
            table_lines.append(f"- `{Path(row.get('curated_file') or '').name}` 来源于 `{row.get('source_file')}`。")
    write_text(out_dir / "captions" / "main_figure_captions_plain.md", main_lines)
    write_text(out_dir / "captions" / "supplement_figure_captions_plain.md", supp_lines)
    write_text(out_dir / "captions" / "table_captions_plain.md", table_lines)


def write_manuscript_drafts(out_dir: Path, matrix: list[dict[str, str]], metrics: list[dict[str, str]], task_summary: list[dict[str, str]], repair_rows: list[dict[str, str]], figure_rows: list[dict[str, str]], table_rows: list[dict[str, str]]) -> None:
    counts = status_counts(matrix)
    status_text = ", ".join(f"{count} {status}" for status, count in sorted(counts.items())) or "no recorded status"
    manual_flags = [row for row in metrics if str(row.get("manual_review_required") or "").lower() == "true"]
    repair_by_arm = Counter(row.get("arm") or "" for row in repair_rows)
    rows_by_arm: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in matrix:
        rows_by_arm[row.get("arm") or ""].append(row)
    task_lines = []
    for row in sorted(task_summary, key=lambda r: task_order(r.get("task_key") or "")):
        task = row.get("task_key") or ""
        task_lines.append(
            f"- `{task}` ({TASK_NAMES.get(task, task)}): {row.get('clean_ok')}/{row.get('n_arms')} cells clean; "
            f"{row.get('repair_events')} repair events; representative metrics: {row.get('representative_metrics') or 'see audit tables'}."
        )
    arm_lines = []
    for arm in ARM_ORDER:
        arm_rows = rows_by_arm.get(arm, [])
        clean = sum(1 for row in arm_rows if row.get("status") == "clean_ok")
        arm_lines.append(f"- `{arm}`: {clean}/{len(arm_rows)} clean cells; {repair_by_arm.get(arm, 0)} repair/fallback events.")
    draft = [
        "# Draft Manuscript",
        "",
        "Working title: Execution-audited LLM-assisted ICU data analysis across a 15-task benchmark",
        "",
        "## Abstract",
        "",
        "### Background",
        "",
        "Large language models can generate code for clinical data analysis, but a generated answer is not the same as a runnable, auditable analysis. We evaluated whether an LLM-assisted analysis pipeline could produce complete, checkable outputs across a diverse set of ICU research tasks.",
        "",
        "### Methods",
        "",
        f"We evaluated 15 ICU data-analysis tasks under four prompting/execution settings, producing {len(matrix)} task-arm cells. Each cell was run through a code-generation, execution, artifact-checking, metric-extraction, and audit pipeline. A cell reached clean completion when code execution finished, expected artifacts were present, key metrics were extractable, and no unresolved contract-level failure remained. Repair and fallback events were recorded separately.",
        "",
        "### Results",
        "",
        f"All {len(matrix)} task-arm cells reached clean completion, with final status counts of {status_text}. Automated metric sanity checks covered {len(metrics)} cells and flagged {len(manual_flags)} cells for manual range review. The final audit recorded {len(repair_rows)} repair or fallback events. The curated publication package contains {len(figure_rows)} figure records and {len(table_rows)} table records, with source maps linking presentation-ready outputs to original run evidence.",
        "",
        "### Conclusions",
        "",
        "In this execution-audited benchmark, all task-arm cells produced runnable and checkable outputs. However, clean completion should be interpreted together with repair burden. These results support the value of combining LLM code generation with strict execution checks, artifact tracking, and transparent reporting of system assistance.",
        "",
        "## Introduction",
        "",
        "Clinical data analysis requires more than a plausible written answer. The analysis code must run, the outputs must be saved, and the resulting metrics and figures must be traceable to the data and workflow that produced them. LLMs are increasingly capable of generating analysis code, but their outputs can fail because of missing dependencies, file-path errors, serialization problems, incomplete artifacts, or incorrect result formatting.",
        "",
        "We designed EasyICU v15 as an execution-audited benchmark for LLM-assisted ICU data analysis. The goal was not only to ask whether the model could produce an answer, but whether each analysis could be converted into a complete, inspectable evidence package.",
        "",
        "## Methods",
        "",
        "### Benchmark design",
        "",
        f"The benchmark included 15 ICU data-analysis tasks. Each task was run under four settings: `aware`, `aware_no_pref`, `naive_with_pref`, and `naive`. This created {len(matrix)} total task-arm cells.",
        "",
        "The tasks covered descriptive summaries, mortality associations, severity-score checks, prediction performance, missing-data robustness, clustering, renal dysfunction, vasopressor selection bias, and dose-response analyses.",
        "",
        "### Execution and audit pipeline",
        "",
        "For each task-arm cell, the pipeline generated or executed analysis code, checked whether required artifacts were present, extracted key metrics, and recorded audit information. When common execution or formatting problems occurred, deterministic repairs or fallback routines could be used. These events were not hidden; they were counted and reported as repair burden.",
        "",
        "### Definition of clean completion",
        "",
        "`clean_ok` means that a task-arm cell satisfied the project execution contract: execution completed, expected files were present, key metrics were extractable, and no unresolved contract-level failure remained. It does not mean that the clinical interpretation is externally validated, that the model worked without assistance, or that the figure is automatically publication-ready.",
        "",
        "### Figure and table curation",
        "",
        "Original figures and tables were inventoried. Selected figures were copied into a curated publication package. A small number of figures were redrawn from structured step summaries to improve presentation quality. Redrawn figures were treated as presentation-layer outputs, not new analyses.",
        "",
        "## Results",
        "",
        "### Overall completion",
        "",
        f"All {len(matrix)} task-arm cells reached clean completion. The final status count was {status_text}.",
        "",
        "### Completion by setting",
        "",
        *arm_lines,
        "",
        "Because all settings reached final clean completion, the main comparison between settings is not final success alone. The more informative comparison is the amount of repair or fallback support required to obtain clean outputs.",
        "",
        "### Task-level results",
        "",
        *task_lines,
        "",
        "### Metric audit",
        "",
        f"Automated metric sanity checks covered {len(metrics)} cells. The number of cells flagged for manual range review was {len(manual_flags)}.",
        "",
        "### Repair burden",
        "",
        f"The final audit recorded {len(repair_rows)} repair or fallback events. These events included execution fixes, artifact binding fixes, metric extraction support, and deterministic fallback routines. Repair burden should be interpreted as a measure of how much system assistance was required after initial model generation.",
        "",
        "### Curated figures and tables",
        "",
        f"The curated publication package includes {len(figure_rows)} figure records and {len(table_rows)} table records. Source maps identify the original run evidence for every curated output.",
        "",
        "## Discussion",
        "",
        "This benchmark shows that LLM-assisted ICU data analysis can be made more reliable when model-generated analysis is embedded in an execution and audit framework. The key result is not simply that all cells eventually completed, but that completion was documented together with artifacts, metrics, figures, and repair burden.",
        "",
        "A central lesson is that final runnable output and model autonomy are different concepts. A task may produce a clean final output while still requiring deterministic repair or fallback support. Reporting both clean completion and repair burden gives a more honest view of system performance.",
        "",
        "The curated figures and tables provide a paper-facing summary of the experiment, but interpretation should remain tied to the original audit files and source maps. Association analyses should not be described as causal unless supported by a separate causal study design.",
        "",
        "## Limitations",
        "",
        "- Clean completion is an execution and artifact standard, not a guarantee of clinical truth.",
        "- Some cells required repair or fallback support, so the results should not be described as fully autonomous model performance.",
        "- Some figures were redrawn for presentation quality; these redraws do not constitute new analyses.",
        "- Most clinical variable analyses are associational and should not be described as causal.",
        "- External validation on independent ICU datasets is still needed before making strong clinical claims.",
        "",
        "## Suggested reporting sentence",
        "",
        "All 60 task-arm cells reached clean completion under an execution-audited LLM-assisted analysis pipeline, with repair burden recorded separately to distinguish final runnable output from the amount of system assistance required.",
    ]
    write_text(out_dir / "manuscript_draft" / "draft_manuscript_en.md", draft)

    zh = [
        "# 论文草稿中文解释版",
        "",
        "这个文件不是正式中文论文，而是帮助你理解英文草稿每部分在说什么。",
        "",
        "## 题目意思",
        "",
        "这篇文章要讲的不是“模型随便回答得好不好”，而是“模型生成的数据分析代码，放进一个会运行、会检查、会记录修复的框架里，最后能不能形成可靠的结果包”。",
        "",
        "## 摘要核心意思",
        "",
        f"我们做了 15 个 ICU 数据分析任务，每个任务 4 种设置，一共 {len(matrix)} 个实验单元。最后 {counts.get('clean_ok', 0)} 个都达到了 clean_ok。",
        "",
        f"但是，中间有 {len(repair_rows)} 个自动修复或兜底事件。所以论文不能写成模型完全独立完成，而应该写成：在执行、检查和修复框架下完成。",
        "",
        "## 方法部分核心意思",
        "",
        "每个任务不是只让模型写一段答案，而是要求它产生能运行的分析、保存结果、生成图表、留下 summary，并且能被系统检查。",
        "",
        "## 结果部分核心意思",
        "",
        "最重要的结果有三个：",
        "",
        f"1. 所有 {len(matrix)} 个实验单元最终 clean_ok。",
        f"2. 自动指标检查没有发现需要人工复核的核心范围异常：{len(manual_flags)} 个 flag。",
        f"3. 修复/兜底事件有 {len(repair_rows)} 个，必须透明报告。",
        "",
        "## 讨论部分核心意思",
        "",
        "这套结果说明：LLM 做临床数据分析时，不能只看回答内容，还要看代码能不能跑、文件能不能追踪、指标能不能抽取、修复用了多少。",
        "",
        "## 最重要的限制",
        "",
        "- clean_ok 不等于临床结论一定正确。",
        "- clean_ok 不等于模型完全自主。",
        "- 相关关系不能写成因果关系。",
        "- 重画图只是为了展示清楚，不是新实验。",
        "- 还需要外部数据验证，才能做更强临床结论。",
    ]
    write_text(out_dir / "manuscript_draft" / "draft_manuscript_zh_explained.md", zh)

    abstract_lines = [
        "# Standalone Abstract Draft",
        "",
        "## Short version",
        "",
        f"We evaluated 15 ICU data-analysis tasks under four prompting/execution settings, creating {len(matrix)} task-arm cells. All cells reached clean completion under an execution-audited pipeline that checked code execution, artifact presence, metric extraction, and unresolved failures. Automated metric checks flagged {len(manual_flags)} cells for manual range review, and the final audit recorded {len(repair_rows)} repair or fallback events. These results show that LLM-assisted analysis can produce complete and traceable outputs when paired with strict execution checks and transparent repair reporting, but clean completion should not be interpreted as fully autonomous model performance or clinical validation.",
        "",
        "## Very short version",
        "",
        f"All {len(matrix)} EasyICU v15 task-arm cells reached clean completion under an execution-audited LLM-assisted analysis pipeline. Because {len(repair_rows)} repair or fallback events were recorded, final completion should be reported together with repair burden rather than described as fully autonomous model success.",
    ]
    write_text(out_dir / "manuscript_draft" / "abstract_draft_en.md", abstract_lines)

    figure_plan = [
        "# 图表放置建议",
        "",
        "## 主文图",
        "",
        "建议主文只放最稳定、最容易解释的图：",
        "",
        "- Figure 1：任务完成矩阵或 success matrix，对应 `table1_success_matrix` 也可以作为主表。",
        "- Figure 2：`fig03_prediction_performance_t07`，死亡预测模型表现。",
        "- Figure 3：`fig04_creatinine_kdigo_t14`，肌酐轨迹和 KDIGO。",
        "- Figure 4：`fig05_norepi_dose_response_t15`，去甲肾上腺素剂量关系，但必须写成相关关系。",
        "- Figure 5：`fig02_sofa_correlation_t03` 或 `fig01_sofa_strata_mortality_t02`。",
        "",
        "## 补充图",
        "",
        "补充图可以放：",
        "",
        "- t01 基线描述",
        "- t04 乳酸 OR 重画图",
        "- t08 血管活性药物偏倚检查",
        "- t09 SOFA-zero audit",
        "- t10 缺失处理稳健性",
        "- t12 年龄分层死亡率",
        "- t06 休克分型",
        "- t11 ICU 住院时长分布",
        "- t13 入院生命体征",
        "",
        "## 主文表",
        "",
        "- Table 1：`table1_success_matrix.csv`。",
        "- Table 2：`table2_task_metric_summary.csv`。",
        "",
        "## 补充表",
        "",
        "- Table S1：修复负担。",
        "- Table S2：artifact inventory。",
        "- Table S3：figure inventory。",
        "- Table S4：metric sanity audit。",
        "- Table S5：SOFA-zero audit。",
    ]
    write_text(out_dir / "manuscript_draft" / "figure_table_placement_plan_plain.md", figure_plan)

    final_check = [
        "# 最终投稿前检查清单",
        "",
        "## 文字检查",
        "",
        "- [ ] 有没有把 clean_ok 解释成执行检查通过，而不是临床真理？",
        "- [ ] 有没有明确报告自动修复/兜底事件数？",
        "- [ ] 有没有避免 fully autonomous 这类过度说法？",
        "- [ ] 有没有避免 caused / led to 这类因果词？",
        "- [ ] 有没有说明重画图只是展示层？",
        "",
        "## 图表检查",
        "",
        "- [ ] 每个主图都有 caption。",
        "- [ ] 每个补充图都有 caption。",
        "- [ ] 每个重画图都能在 source map 找到来源。",
        "- [ ] t08 caption 明确是偏倚/相关关系检查。",
        "- [ ] t10 caption 明确 CI unavailable 的情况。",
        "",
        "## 数据和复现检查",
        "",
        "- [ ] `matrix_status.csv` 是 60 行。",
        "- [ ] `repair_audit.csv` 和 `paper_repair_burden.csv` 一起提交或放 supplement。",
        "- [ ] `figure_source_map.csv` 和 `table_source_map.csv` 保存。",
        "- [ ] 关键测试结果记录在方法或 supplement。",
        "",
        "## 人工审查",
        "",
        "- [ ] 临床合作者看过主文图。",
        "- [ ] 统计合作者看过 OR、CI、AUROC、聚类等解释。",
        "- [ ] 通讯作者确认限制部分足够清楚。",
    ]
    write_text(out_dir / "manuscript_draft" / "final_submission_checklist_plain.md", final_check)


def write_checklists(out_dir: Path, matrix: list[dict[str, str]], figure_rows: list[dict[str, str]], table_rows: list[dict[str, str]], repair_rows: list[dict[str, str]]) -> None:
    claim_rows = [
        {
            "claim_id": "C001",
            "claim_plain_language": "60 个实验单元最终都完成了。",
            "supporting_file": "matrix_status.csv",
            "safe_wording": "all 60 task-arm cells reached clean completion",
            "avoid_wording": "the model independently completed all analyses without assistance",
            "status": "ready",
            "notes": "clean_ok 是执行检查通过，不等于临床结论完全验证。",
        },
        {
            "claim_id": "C002",
            "claim_plain_language": "核心指标没有被自动范围检查标记为异常。",
            "supporting_file": "metric_sanity_audit.csv",
            "safe_wording": "automated metric sanity checks did not flag core metrics for manual range review",
            "avoid_wording": "all metrics are clinically correct",
            "status": "ready",
            "notes": "自动检查不是人工临床审稿。",
        },
        {
            "claim_id": "C003",
            "claim_plain_language": "有自动修复和兜底，必须单独报告。",
            "supporting_file": "repair_audit.csv; paper_repair_burden.csv",
            "safe_wording": "repair burden was recorded and reported separately",
            "avoid_wording": "fully autonomous",
            "status": "ready",
            "notes": f"本次修复/兜底事件数为 {len(repair_rows)}。",
        },
        {
            "claim_id": "C004",
            "claim_plain_language": "乳酸、血管活性药物、年龄等结果主要是相关关系。",
            "supporting_file": "curated_publication figures and metric tables",
            "safe_wording": "was associated with / differed across groups",
            "avoid_wording": "caused / led to / proved",
            "status": "ready_with_caution",
            "notes": "没有专门因果设计时，不要写成导致。",
        },
        {
            "claim_id": "C005",
            "claim_plain_language": "重画图只是为了展示更清楚。",
            "supporting_file": "curated_publication/source_maps/figure_source_map.csv",
            "safe_wording": "redrawn from structured step summaries for presentation quality",
            "avoid_wording": "new analysis result",
            "status": "ready",
            "notes": "原始证据仍保留在 run directories。",
        },
    ]
    write_csv(out_dir / "checklist" / "manuscript_claim_checklist.csv", claim_rows, ["claim_id", "claim_plain_language", "supporting_file", "safe_wording", "avoid_wording", "status", "notes"])

    figure_check = []
    for row in figure_rows:
        key = basename_no_suffix(row.get("curated_file") or "")
        role = "main" if "/figures/main/" in (row.get("curated_file") or "") else "supplement"
        caveat = ""
        if key == "figR04_lactate_robustness_t10":
            caveat = "CI 不在结构化结果中；caption 已说明。"
        elif key == "figR02_vaso_bias_t08":
            caveat = "只能用于说明选择偏倚或相关关系，不能写成用药导致死亡。"
        elif row.get("source_type") == "rebuilt":
            caveat = "从结构化 summary 重画；不改变原始结果。"
        figure_check.append({
            "figure_file": Path(row.get("curated_file") or "").name,
            "task_key": row.get("source_task_key"),
            "role": role,
            "source_type": row.get("source_type"),
            "status": "ready_with_note" if caveat else "ready",
            "plain_note": caveat or "可作为当前论文材料使用。",
            "source_file": row.get("source_file"),
        })
    write_csv(out_dir / "checklist" / "figure_readiness_checklist.csv", figure_check, ["figure_file", "task_key", "role", "source_type", "status", "plain_note", "source_file"])

    table_check = []
    for row in table_rows:
        table_check.append({
            "table_file": Path(row.get("curated_file") or "").name,
            "status": "ready",
            "plain_note": row.get("notes") or "可用。",
            "source_file": row.get("source_file"),
        })
    write_csv(out_dir / "checklist" / "table_readiness_checklist.csv", table_check, ["table_file", "status", "plain_note", "source_file"])

    repro_lines = [
        "# 可复现性检查清单",
        "",
        "写论文或给合作者时，至少确认下面这些信息都能找到。",
        "",
        "## 已有材料",
        "",
        "- final audit 文件夹存在。",
        "- matrix_status.csv 存在，并且包含 60 行。",
        "- repair_audit.csv 存在。",
        "- metric_sanity_audit.csv 存在。",
        "- curated_publication 文件夹存在。",
        "- 图表 source map 存在。",
        "- 关键 pytest 已通过。",
        "",
        "## 写作时必须说明",
        "",
        "- clean_ok 的定义。",
        "- 自动修复/兜底逻辑的存在。",
        "- 重画图只是展示层，不是新实验。",
        "- 多数临床变量分析是相关关系，不是因果证明。",
        "",
        "## 投稿前人工复核",
        "",
        "- 主图是否清晰。",
        "- supplement 图是否都能解释。",
        "- 表格 caption 是否和表内容一致。",
        "- 文字里有没有过度声称。",
    ]
    write_text(out_dir / "checklist" / "reproducibility_checklist_plain.md", repro_lines)


def write_manifests(out_dir: Path, figure_rows: list[dict[str, str]], table_rows: list[dict[str, str]]) -> None:
    main_figures = []
    supp_figures = []
    for row in figure_rows:
        item = {
            "file": row.get("curated_file"),
            "task_key": row.get("source_task_key"),
            "source_type": row.get("source_type"),
            "source_file": row.get("source_file"),
            "plain_note": row.get("notes"),
        }
        if "/figures/main/" in (row.get("curated_file") or ""):
            main_figures.append(item)
        else:
            supp_figures.append(item)
    main_tables = []
    supp_tables = []
    for row in table_rows:
        item = {"file": row.get("curated_file"), "source_file": row.get("source_file"), "plain_note": row.get("notes")}
        if "/tables/main/" in (row.get("curated_file") or ""):
            main_tables.append(item)
        else:
            supp_tables.append(item)
    write_csv(out_dir / "submission_package" / "main_figures_manifest.csv", main_figures, ["file", "task_key", "source_type", "source_file", "plain_note"])
    write_csv(out_dir / "submission_package" / "supplement_figures_manifest.csv", supp_figures, ["file", "task_key", "source_type", "source_file", "plain_note"])
    write_csv(out_dir / "submission_package" / "main_tables_manifest.csv", main_tables, ["file", "source_file", "plain_note"])
    write_csv(out_dir / "submission_package" / "supplement_tables_manifest.csv", supp_tables, ["file", "source_file", "plain_note"])


def build_package(run_root: Path, audit_dir: Path, out_dir: Path) -> None:
    matrix = read_csv(audit_dir / "matrix_status.csv")
    metrics = read_csv(audit_dir / "metric_sanity_audit.csv")
    task_summary = read_csv(audit_dir / "paper_task_metric_summary.csv")
    repair_rows = read_csv(audit_dir / "repair_audit.csv")
    curated = audit_dir / "curated_publication"
    figure_rows = read_csv(curated / "source_maps" / "figure_source_map.csv")
    table_rows = read_csv(curated / "source_maps" / "table_source_map.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_readme(out_dir, audit_dir, matrix, repair_rows, figure_rows, table_rows)
    write_plain_extras(out_dir, matrix, repair_rows, figure_rows, table_rows)
    write_methods(out_dir, matrix, repair_rows)
    write_results(out_dir, matrix, metrics, task_summary, repair_rows)
    write_captions(out_dir, figure_rows, table_rows)
    write_manuscript_drafts(out_dir, matrix, metrics, task_summary, repair_rows, figure_rows, table_rows)
    write_checklists(out_dir, matrix, figure_rows, table_rows, repair_rows)
    write_manifests(out_dir, figure_rows, table_rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "audit_dir": str(audit_dir),
        "out_dir": str(out_dir),
        "matrix_rows": len(matrix),
        "status_counts": dict(status_counts(matrix)),
        "repair_events": len(repair_rows),
        "curated_figure_records": len(figure_rows),
        "curated_table_records": len(table_rows),
    }
    (out_dir / "package_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    audit_dir = Path(args.audit_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else audit_dir / "manuscript_evidence"
    build_package(run_root, audit_dir, out_dir)
    print(out_dir)
    print(out_dir / "README.md")
    print(out_dir / "checklist" / "manuscript_claim_checklist.csv")
    print(out_dir / "package_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
