"""Public, method-first Skill Hub projection of real EasyICU capabilities.

The research-agent registries own execution and scientific authority.  This
module owns only the user-facing catalogue: a small set of discoverable method
workflows whose capability and action identifiers are checked at import time.
It also projects a read-only package for the desktop Skill Hub, separating the
main instruction, reference contracts, validation rules, and any real
capability-specific helpers. Read-only host excerpts disclose the computing
definitions without granting execution authority to the package.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Tuple

from easyicu.databases.profiles import get_database_profile, public_database_keys
from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.research_requirements import StudyRequirements, ThresholdRequirement, assess_study

from .planning.capability_registry import CAPABILITY_REGISTRY
from .planning.analysis_method_suite import METHOD_SUITE_REGISTRY
from .planning.method_adapter_catalog import HIGH_FREQUENCY_METHOD_ADAPTERS
from .skill_packages.host_sources import method_skill_implementation_files

METHOD_SKILL_REGISTRY_VERSION = "easyicu.method-skills/3"
METHOD_SKILL_PACKAGE_VERSION = "easyicu.method-skill-package/5"

#: Workflows whose package ships an executable reference implementation from
#: ``research_agent/skill_packages/<package>``: real ``scripts/`` that run the
#: standard workflow through the host kernels, plus the package's own SKILL.md
#: and method notes. Other workflows still require host execution.
REFERENCE_SCRIPT_PACKAGES: dict[str, str] = {
    "fixed-landmark-association-study": "landmark_categorical_association",
    "cohort-characterization-table-one": "cohort_characterization_table_one",
    "adjusted-association-model": "adjusted_association_model",
    "survival-time-to-event": "survival_time_to_event",
}
DOCUMENTATION_ONLY_BOUNDARY = "documentation_only_host_execution_required"
REFERENCE_SCRIPTS_BOUNDARY = "host_reference_scripts_executable"

ClaimCeiling = Literal["reportable", "analysis_only"]
ExecutionMode = Literal["deterministic_host", "agent_coded_with_host_gates"]
SkillLayer = Literal["research_workflow", "analysis_module", "data_definition", "planning_aid"]


OWNER_ENTRIES = {
    "six-database-cohort-extraction": ("data_definition", "data_extraction", "/api/jobs/extract"),
    "clinical-concept-definitions": ("data_definition", "concept_catalog",
                                     "easyicu.research_agent.concept_catalog.clinical_definitions"),
    "sample-size-and-power": ("planning_aid", "sample_size",
                              "easyicu.research_agent.skill_packages"),
}


@dataclass(frozen=True, slots=True)
class OwnerSkillSpec:
    skill_id: str
    title: str
    title_zh: str
    description: str
    description_zh: str
    layer: Literal["data_definition", "planning_aid"]
    category: str
    category_zh: str
    owner: str
    entry: str
    claim_ceiling: str | None = None
    capability_id: None = None
    action_ids: tuple[str, ...] = ()
    included_module_ids: tuple[str, ...] = ()

    def to_dict(self, *, enabled: bool) -> dict[str, Any]:
        validate_owner_skill(self)
        prefill: dict[str, Any] = {}
        if self.owner == "data_extraction":
            from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL

            prefill = {"modules": list(CONCEPT_GROUPS_INTERNAL), "format": "parquet",
                       "merge": True, "include_feature_definitions": True}
        row = {
            "id": self.skill_id, "title": self.title, "title_zh": self.title_zh,
            "description": self.description, "description_zh": self.description_zh,
            "category": self.category, "category_zh": self.category_zh,
            "layer": self.layer, "enabled": bool(enabled), "version": "easyicu.clinician-skills/1",
            "capability_id": None, "action_ids": [], "included_module_ids": [],
            "workflow_steps": [], "execution_mode": (
                "owner_projection" if self.layer == "data_definition" else "deterministic_host"
            ),
            "scope": self.description,
            "prompt": f"Use {self.title}. My research question: ",
            "prompt_zh": f"请使用“{self.title_zh}”。我的研究问题是：",
            "use_action": {"kind": "open_owner", "owner": self.owner, "entry": self.entry,
                           "enabled": bool(enabled), "auto_submit": False, "prefill": prefill},
        }
        if self.claim_ceiling is not None:
            row["claim_ceiling"] = self.claim_ceiling
        return row


def validate_owner_skill(row: OwnerSkillSpec) -> None:
    expected = OWNER_ENTRIES.get(row.skill_id)
    if expected != (row.layer, row.owner, row.entry):
        raise RuntimeError("skill.owner_binding_invalid")
    if row.capability_id is not None or row.action_ids or row.included_module_ids:
        raise RuntimeError("skill.projection_cannot_execute_analysis")
    if row.layer == "data_definition" and row.claim_ceiling is not None:
        raise RuntimeError("skill.definition_cannot_claim")
    if row.layer == "planning_aid" and row.claim_ceiling != "analysis_only":
        raise RuntimeError("skill.planning_analysis_only")


OWNER_SKILLS = (
    OwnerSkillSpec(
        "six-database-cohort-extraction", "Six-database cohort extraction", "六数据库队列提取",
        "Choose a database, cohort and variables to export through Data Extraction.",
        "选择数据库、队列和变量，通过数据提取导出。", "data_definition", "Data and definitions",
        "数据与定义", "data_extraction", "/api/jobs/extract",
    ),
    OwnerSkillSpec(
        "clinical-concept-definitions", "Clinical concept definitions", "临床概念定义",
        "Inspect score components, time windows and database availability before choosing a definition.",
        "选择定义前查看评分组成、时间窗和各数据库的可用性。", "data_definition", "Data and definitions",
        "数据与定义", "concept_catalog", OWNER_ENTRIES["clinical-concept-definitions"][2],
    ),
    OwnerSkillSpec(
        "sample-size-and-power", "Sample size and power", "样本量与效能",
        "Estimate participants or events needed from your study assumptions before analysis.",
        "分析前根据研究假设估算所需人数或事件数。", "planning_aid", "Study planning", "研究规划",
        "sample_size", OWNER_ENTRIES["sample-size-and-power"][2], "analysis_only",
    ),
)


@dataclass(frozen=True, slots=True)
class MethodSkillSpec:
    """One discoverable workflow or reusable module backed by registered owners."""

    skill_id: str
    title: str
    title_zh: str
    category: str
    category_zh: str
    description: str
    description_zh: str
    capability_id: str
    action_ids: Tuple[str, ...]
    execution_mode: ExecutionMode
    claim_ceiling: ClaimCeiling
    prompt: str
    prompt_zh: str
    layer: SkillLayer = "research_workflow"
    included_module_ids: Tuple[str, ...] = ()
    workflow_steps: Tuple[str, ...] = ()
    not_for: tuple[tuple[str, str, str | None], ...] = ()

    def to_dict(self, *, enabled: bool) -> dict[str, Any]:
        capability = _CAPABILITY_BY_ID[self.capability_id]
        return {
            "id": self.skill_id,
            "title": self.title,
            "title_zh": self.title_zh,
            "category": self.category,
            "category_zh": self.category_zh,
            "description": self.description,
            "description_zh": self.description_zh,
            "version": METHOD_SKILL_REGISTRY_VERSION,
            "enabled": bool(enabled),
            "capability_id": self.capability_id,
            "action_ids": list(self.action_ids),
            "layer": self.layer,
            "included_module_ids": list(self.included_module_ids),
            "workflow_steps": list(self.workflow_steps),
            "not_for": [dict(situation_en=en, situation_zh=zh, sibling_skill_id=sibling)
                        for en, zh, sibling in self.not_for],
            "execution_mode": self.execution_mode,
            "claim_ceiling": self.claim_ceiling,
            "scope": capability.primary_estimand,
            "inputs": list(capability.data_contract),
            "outputs": [capability.result_contract],
            "diagnostics": list(capability.required_diagnostics),
            "prompt": self.prompt,
            "prompt_zh": self.prompt_zh,
        }


METHOD_SKILLS: Tuple[MethodSkillSpec | OwnerSkillSpec, ...] = (
    MethodSkillSpec(
        "repeated-measure-trajectory", "Repeated-measure trajectory study", "重复测量轨迹研究",
        "Longitudinal analysis", "纵向分析",
        "Estimate changes in repeated ICU measurements with mixed models or GEE, with explicit death-dropout limitations.",
        "用混合效应模型或 GEE 估计 ICU 重复测量的时间变化，明确死亡导致的信息性脱落局限。",
        "repeated_measures_v1", ("association.mixed_effects", "association.gee"),
        "deterministic_host", "analysis_only",
        "Use the Repeated-measure trajectory study. Review ICU time zero, hours/days, window, aggregation, mixed model or GEE working correlation, and informative death-dropout sensitivity before execution. My research question: ",
        "请使用“重复测量轨迹研究”。执行前审阅 ICU 时间零点、小时或天、窗口、聚合方式、混合模型或 GEE 工作相关结构，以及死亡所致信息性脱落的敏感性分析。我的研究问题是：",
        workflow_steps=("Audit the source long panel and measurement density.",
                        "Freeze ICU time origin, unit, window, bin width, aggregation and fixed-time basis.",
                        "Fit the reviewed random-intercept/time-slope model or robust GEE.",
                        "Restrict to survivors as a sensitivity or state a joint-model note; retain the death-dropout limitation.",
                        "Report aggregate effects, covariance/ICC or QIC, and observed time profiles."),
        not_for=(("Discover latent trajectory classes", "发现潜在轨迹类别", "trajectory-phenotyping"),),
    ),
    MethodSkillSpec(
        "treatment-patterns", "Treatment patterns", "治疗使用模式", "Descriptive", "描述性研究",
        "Describe drug-class use during the ICU stay and time to first recorded use by group.",
        "按分组描述 ICU 期间某类药物的使用比例及首次记录用药的时间。",
        "descriptive_measurement_v1", ("descriptive.table_one", "time_to_event.competing_risks_cif"),
        "deterministic_host", "analysis_only",
        "Use Treatment patterns with reviewed class capture, groups and ICU time origin. My research question: ",
        "请使用“治疗使用模式”，在计划中审阅药物覆盖、分组和 ICU 时间起点。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "cohort-characterization-table-one",
        "Cohort characterization and Table 1",
        "队列描述与 Table 1",
        "Descriptive",
        "描述性研究",
        "Describe the study population's baseline characteristics by group as a Table 1, with each variable's true denominator.",
        "按分组描述研究人群的基线特征，生成 Table 1，并保留每个变量的真实分母。",
        "descriptive_measurement_v1",
        ("descriptive.table_one",),
        "deterministic_host",
        "analysis_only",
        "Use the Cohort characterization and Table 1 method. The research plan proposes the grouping levels, summarized variables, units, and summary rules; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“队列描述与 Table 1”方法。分组水平、汇总变量、单位与汇总规则由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "missingness-measurement-audit",
        "Missingness and measurement audit",
        "缺失与测量审计",
        "Descriptive",
        "描述性研究",
        "Check, variable by variable, how many patients have a measurement, how much is missing and in which time window, before choosing how to handle missing data.",
        "逐个变量查看有多少患者有记录、缺失多少、在哪个时间窗测得，再决定如何处理缺失。",
        "descriptive_measurement_v1",
        ("descriptive.missingness_audit", "association.missingness_audit"),
        "deterministic_host",
        "analysis_only",
        "Use the Missingness and measurement audit method. The research plan proposes the audited variables, time windows, denominators, and structural-absence rules; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“缺失与测量审计”方法。审计变量、时间窗、分母与结构性缺失规则由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "exposure-outcome-distribution",
        "Exposure and outcome distribution",
        "暴露与结局分布",
        "Descriptive",
        "描述性研究",
        "Report outcome risks and denominators for each exposure group; descriptive only, with no causal wording.",
        "按暴露分组计算结局的发生率和分母，只做描述，不作因果解释。",
        "descriptive_exposure_outcome_distribution_v1",
        (),
        "deterministic_host",
        "reportable",
        "Use the Exposure and outcome distribution method. The research plan proposes the exposure levels, outcome definition, denominators, and interval method; I review them together in the plan, and nothing runs before that review. Interpretation stays descriptive. My research question: ",
        "请使用“暴露与结局分布”方法。暴露水平、结局定义、分母与区间方法由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。解释保持描述性，不作因果推断。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "adjusted-association-model",
        "Adjusted association model",
        "调整后关联模型",
        "Association",
        "关联分析",
        "Estimate the exposure–outcome association adjusted for confounders with one regression model; variable coding and reference groups are stated in the plan.",
        "用一个回归模型估计调整混杂后的暴露—结局关联；变量编码和参照组在计划中写明。",
        "association_adjusted_v1",
        ("association.adjusted_association",),
        "deterministic_host",
        "reportable",
        "Use the Adjusted association model method. The research plan proposes the estimand, covariates, coding and reference level, analysis population, and missing-data strategy; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“调整后关联模型”方法。估计量、协变量、编码与参照水平、分析人群和缺失数据策略由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "ordinal-dose-response",
        "Ordinal trend and dose-response",
        "有序趋势与剂量反应",
        "Association",
        "关联分析",
        "When an exposure has at least three ordered levels (e.g. low, medium, high), test whether the outcome changes steadily across them.",
        "暴露分为至少三个有顺序的等级（如低、中、高）时，检验结局是否随等级逐步变化。",
        "association_ordinal_trend_v1",
        ("association.ordinal_trend",),
        "agent_coded_with_host_gates",
        "analysis_only",
        "Use the Ordinal trend and dose-response method. The research plan proposes the ordered levels and reference, primary contrast, adjustment set, and trend test; I review them together in the plan, and nothing runs before that review. A binary or continuous exposure is not forced into ordered groups. My research question: ",
        "请使用“有序趋势与剂量反应”方法。有序水平与参照、主要对比、调整集与趋势检验由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。不会把二分类或连续暴露改成有序分组。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "landmark-categorical-association",
        "Landmark categorical association",
        "Landmark 分类关联",
        "Association",
        "关联分析",
        "Among patients still in hospital at a set time point (e.g. 24 h after ICU admission), compare outcomes across exposure groups without immortal-time bias.",
        "在某个时间点（如入 ICU 后 24 小时）仍在院的患者中，比较不同暴露分组的结局，避免不死时间偏倚。",
        "association_landmark_categorical_v1",
        ("association.adjusted_association",),
        "deterministic_host",
        "reportable",
        "Use the Landmark categorical association method. The research plan proposes time zero, the landmark, the eligible population, exposure levels and reference, and the adjustment set; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“Landmark 分类关联”方法。时间零点、landmark、合格人群、暴露水平与参照、调整集由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "landmark-spline-dose-response",
        "Landmark spline dose-response",
        "Landmark 样条剂量反应",
        "Association",
        "关联分析",
        "Among patients still in hospital at a set time point, describe the non-linear relation between a continuous exposure (e.g. peak lactate) and the outcome with a spline.",
        "在固定时间点仍在院的患者中，用样条曲线描述连续暴露（如最高乳酸）与结局的非线性关系。",
        "association_landmark_spline_v1",
        (),
        "deterministic_host",
        "reportable",
        "Use the Landmark spline dose-response method. The research plan proposes time zero, the landmark, the exposure window, spline knots and reference value, and the adjustment set; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“Landmark 样条剂量反应”方法。时间零点、landmark、暴露窗口、样条结点与参考值、调整集由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "time-varying-cox",
        "Time-varying exposure Cox model",
        "随时间变化暴露的 Cox 模型",
        "Survival",
        "生存分析",
        "When the exposure changes during the stay (e.g. on or off vasopressors), update it over time and fit a Cox model.",
        "暴露在住院期间会变化（如是否在用血管活性药）时，按时间分段更新暴露，拟合 Cox 模型。",
        "association_time_varying_exposure_v1",
        ("causal_emulation.g_methods",),
        "deterministic_host",
        "analysis_only",
        "Use the Time-varying exposure Cox model method. The research plan proposes time zero, interval construction, exposure-update rules, event and censoring rules, and covariates; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“随时间变化暴露的 Cox 模型”方法。时间零点、区间构造、暴露更新规则、事件与删失规则和协变量由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "adjusted-exposure-outcome-study",
        "Exposure–outcome association study",
        "暴露—结局关联研究",
        "Association",
        "关联分析",
        "A complete study of whether an exposure is associated with an outcome: cohort description, missingness audit, crude risks and the adjusted primary model.",
        "完整回答“某个暴露是否与结局相关”：从队列描述、缺失审计、粗率比较到调整后的主要模型。",
        "association_adjusted_v1",
        ("association.adjusted_association",),
        "deterministic_host",
        "reportable",
        "Use the Exposure–outcome association study method. The research plan proposes the cohort, time zero, covariates, missing-data strategy, diagnostics, and sensitivity analyses; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“暴露—结局关联研究”方法。队列、时间零点、协变量、缺失数据策略、诊断与敏感性分析由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        not_for=(
            ('A treatment effect, rather than an association', '问题是治疗的因果效应', 'target-trial-emulation'),
            ('Exposure is measured after admission', '暴露需要在入院后的一段时间内测量', 'fixed-landmark-association-study'),
        ),
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
            "exposure-outcome-distribution",
        ),
        workflow_steps=(
            "Freeze the cohort, exposure, outcome, time zero, adjustment set, estimand, and analysis population.",
            "Describe the cohort with Table 1 and audit variable availability, denominators, and missingness.",
            "Report exposure prevalence and outcome absolute risks before fitting the primary adjusted model.",
            "Fit the exact typed adjusted model with prespecified coding, references, transforms, and dependence structure.",
            "Run the registered diagnostics and sensitivity analyses, then reconcile every estimate with the frozen plan.",
            "Publish the descriptive and adjusted products with separate wording and an evidence-bound claim ceiling.",
        ),
    ),
    MethodSkillSpec(
        "ordinal-dose-response-study",
        "Ordered exposure dose-response study",
        "有序暴露剂量反应研究",
        "Association",
        "关联分析",
        "A complete study of an exposure with three or more ordered levels and the outcome: description, adjusted trend test and non-linearity checks, with analysis code written by the AI from the reviewed plan.",
        "完整研究一个分为三级以上的暴露与结局的趋势关系，包括描述、调整后的趋势检验和非线性检查；分析代码由 AI 按审阅后的计划编写。",
        "association_ordinal_trend_v1",
        ("association.ordinal_trend",),
        "agent_coded_with_host_gates",
        "analysis_only",
        "Use the Ordered exposure dose-response study method. The research plan proposes the ordered levels and reference, population, covariates, trend contrast, and nonlinearity checks; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“有序暴露剂量反应研究”方法。有序水平与参照、人群、协变量、趋势对比与非线性检查由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        not_for=(
            ('The exposure is continuous rather than ordered categories', '暴露是连续数值，不是有序分类', 'landmark-spline-dose-response'),
            ('There are only two exposure groups', '暴露仅分为两组', 'adjusted-exposure-outcome-study'),
        ),
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
            "exposure-outcome-distribution",
        ),
        workflow_steps=(
            "Freeze at least three clinically meaningful ordered exposure levels without data-driven regrouping.",
            "Describe the cohort by level and audit measurement density, missingness, and level occupancy.",
            "Report absolute outcome risks and denominators for every closed exposure level.",
            "Estimate the prespecified adjusted trend and the primary level contrast under the declared model contract.",
            "Check sparse levels, model form, nonlinearity, and sensitivity to the declared level coding.",
            "Keep the result analysis-only unless the enclosing study obtains a higher reviewed claim ceiling.",
        ),
    ),
    MethodSkillSpec(
        "fixed-landmark-association-study",
        "Fixed-landmark association study",
        "固定 Landmark 关联研究",
        "Association",
        "关联分析",
        "A complete landmark study: compare outcomes across exposure groups among patients still in hospital at a set time point, without immortal-time bias.",
        "完整的 landmark 研究：在固定时间点仍在院的患者中比较不同暴露分组的结局，避免不死时间偏倚。",
        "association_landmark_categorical_v1",
        ("association.adjusted_association",),
        "deterministic_host",
        "reportable",
        "Use the Fixed-landmark association study method. The research plan proposes time zero, the landmark, the population still eligible at it, outcome horizon, adjustment set, and censoring; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“固定 Landmark 关联研究”方法。时间零点、landmark、届时仍合格的人群、结局时间范围、调整集与删失由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        not_for=(
            ('Exposure keeps changing after the selected time point', '暴露在所选时间点之后仍在变化', 'time-varying-exposure-survival-study'),
            ('The question asks for a treatment effect', '问题要求估计治疗的因果效应', 'target-trial-emulation'),
        ),
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
            "exposure-outcome-distribution",
        ),
        workflow_steps=(
            "Freeze time zero, the landmark, exposure ascertainment window, outcome horizon, and landmark eligibility rule.",
            "Exclude post-landmark information from exposure construction and verify equal observation opportunity.",
            "Describe the landmark-eligible cohort and audit exposure measurement and missingness before modelling.",
            "Report categorical exposure frequencies and outcome absolute risks in the landmark population.",
            "Fit the typed adjusted landmark model with prespecified levels, reference, covariates, censoring, and dependence.",
            "Audit temporal ordering and reconcile the final estimate with the frozen landmark population and evidence receipts.",
        ),
    ),
    MethodSkillSpec(
        "time-varying-exposure-survival-study",
        "Time-varying exposure survival study",
        "随时间变化暴露的生存研究",
        "Survival",
        "生存分析",
        "A complete survival study for an exposure that changes during the stay (e.g. ventilation or a drug), updated over time, with robustness checks.",
        "完整研究住院期间会变化的暴露（如机械通气、用药）与生存的关系，按时间更新暴露并做稳健性检查。",
        "association_time_varying_exposure_v1",
        (),
        "deterministic_host",
        "analysis_only",
        "Use the Time-varying exposure survival study method. The research plan proposes time zero, interval construction, exposure-update rules, event and censoring definitions, patient clustering, and sensitivity analyses; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“随时间变化暴露的生存研究”方法。时间零点、区间构造、暴露更新规则、事件与删失定义、患者聚类和敏感性分析由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        not_for=(
            ('Exposure is fixed before follow-up', '暴露在随访开始之前已经确定', 'survival-time-to-event'),
            ('Time-dependent confounding must be addressed causally', '需要从因果角度处理随时间变化的混杂', 'target-trial-emulation'),
        ),
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
        ),
        workflow_steps=(
            "Freeze time zero, event and censoring rules, exposure updates, interval boundaries, and the analysis horizon.",
            "Describe baseline cohort characteristics and audit longitudinal measurement availability and unmeasured states.",
            "Construct non-overlapping counting-process intervals without carrying future information backward.",
            "Fit the source-bound time-varying Cox model with declared baseline coding and patient-clustered covariance.",
            "Check retained early events, interval coverage, convergence, finite covariance, and alternative update rules.",
            "Report the association as analysis-only and preserve all interval-construction and diagnostic receipts.",
        ),
    ),
    MethodSkillSpec(
        "survival-time-to-event",
        "Survival analysis (time to event)",
        "生存分析（时间到事件）",
        "Survival",
        "生存分析",
        "Compare survival across groups: Kaplan–Meier curves, log-rank test, adjusted Cox hazard ratios and a proportional-hazards check.",
        "比较不同分组的生存：Kaplan–Meier 曲线、log-rank 检验、调整后的 Cox 风险比，并检查比例风险假设。",
        "survival_time_to_event_v1",
        ("time_to_event.cox_hr", "time_to_event.km_logrank", "time_to_event.ph_check"),
        "deterministic_host",
        "reportable",
        "Use the Survival analysis (time to event) method. The research plan proposes the time origin, event and censoring rules, reference level, covariates, follow-up horizon, and proportional-hazards check; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“生存分析（时间到事件）”方法。时间起点、事件与删失规则、参照水平、协变量、观察期限与比例风险检验由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        not_for=(
            ('Exposure changes during follow-up', '暴露在随访期间变化', 'time-varying-exposure-survival-study'),
            ('Only a binary outcome is known, without reliable event times', '只有二分类结局，缺少可靠的事件时间', 'adjusted-exposure-outcome-study'),
        ),
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
        ),
    ),
    MethodSkillSpec(
        "restricted-mean-survival-time",
        "Restricted mean survival time",
        "限制性平均生存时间（RMST）",
        "Survival",
        "生存分析",
        "Compare two groups' mean event-free time within a set follow-up period; interpretable even when hazards are not proportional.",
        "在设定的随访期内比较两组的平均无事件时间；比例风险不成立时也能解释。",
        "survival_time_to_event_v1",
        ("time_to_event.rmst",),
        "deterministic_host",
        "analysis_only",
        "Use the Restricted mean survival time method. The research plan proposes the groups and reference, time origin, event and censoring definitions, and restriction time; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“限制性平均生存时间（RMST）”方法。分组与参照、时间起点、事件与删失定义、限制时间由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "clinical-risk-prediction",
        "Clinical risk prediction model",
        "临床风险预测模型",
        "Prediction",
        "预测模型",
        "Develop and internally validate a binary risk model, with optional frozen-model external validation in other ICU databases; report discrimination, calibration, decision curves and case mix.",
        "开发并内部验证二分类风险模型；可选在其他 ICU 数据库中验证冻结模型，报告区分度、校准、决策曲线和病例构成差异。",
        "prediction_risk_model_v1",
        ("prediction.discrimination_calibration", "prediction.calibration_metrics", "prediction.decision_curve", "prediction.internal_validation", "prediction.external_validation"),
        "deterministic_host",
        "reportable",
        "Use the Clinical risk prediction model method. The research plan proposes the prediction time, horizon, predictors, patient-level split, validation, and calibration; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“临床风险预测模型”方法。预测时点、时间范围、预测变量、患者级数据划分、验证与校准由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        not_for=(
            ('Risk must be updated at several time points', '需要在多个时间点更新风险', 'dynamic-landmark-prediction'),
            ('The question asks whether an exposure is associated with an outcome', '问题是暴露是否与结局相关', 'adjusted-exposure-outcome-study'),
        ),
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
        ),
        workflow_steps=(
            "Develop and internally validate the prespecified binary risk model using patient-level splits.",
            "Optionally review the development database, validation databases, fitted model digest and complete predictor roster for external validation.",
            "Freeze model coefficients and development-fitted preprocessing before reading validation cohorts; verify source capabilities and confirm changed meanings in the plan.",
            "Report untouched transport per database with DeLong AUROC intervals, calibration, Brier, decision curves and case mix; report any apparent recalibration separately, with an analysis-only ceiling.",
        ),
    ),
    MethodSkillSpec(
        "dynamic-landmark-prediction",
        "Dynamic landmark prediction",
        "动态 Landmark 预测",
        "Prediction",
        "预测模型",
        "Update a patient's future risk at several time points after ICU admission from the measurements available so far, for early warning.",
        "在入 ICU 后的多个时间点，用此前的测量更新未来风险，用于早期预警。",
        "dynamic_prediction_landmark_v1",
        (),
        "agent_coded_with_host_gates",
        "analysis_only",
        "Use the Dynamic landmark prediction method. The research plan proposes the landmarks, look-back window, horizon, predictors, and patient-level split; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“动态 Landmark 预测”方法。landmark、回看窗口、时间范围、预测变量与患者级数据划分由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        not_for=(
            ('Only one prediction time is needed', '只需要一个预测时点', 'clinical-risk-prediction'),
            ('The aim is to discover patterns over time', '目的是发现随时间变化的模式', 'trajectory-phenotyping'),
        ),
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
        ),
    ),
    MethodSkillSpec(
        "target-trial-emulation",
        "Target-trial emulation",
        "目标试验模拟（因果推断）",
        "Causal inference",
        "因果推断",
        "State a hypothetical randomized trial (eligibility, treatment strategies, time zero, follow-up), then emulate it with observational data to estimate a treatment's causal effect. Starting a treatment within a grace period versus not starting it is computed by EasyICU's own clone-censor-weight estimator.",
        "先写明一个假想的随机试验（纳入标准、治疗策略、时间零点、随访），再用观察数据模拟它，估计治疗的因果效应。比较宽限期内开始与不开始某治疗时，由 EasyICU 用克隆-删失-加权方法直接计算。",
        "causal_target_trial_v1",
        ("causal_emulation.g_methods",),
        "agent_coded_with_host_gates",
        "analysis_only",
        "Use the Target-trial emulation method. The research plan proposes eligibility, treatment strategies, time zero, follow-up, causal contrast, and confounding control; I review them together in the plan, and nothing runs before that review. Results stay analysis-only unless the identification and readiness gates pass. My research question: ",
        "请使用“目标试验模拟（因果推断）”方法。纳入标准、治疗策略、时间零点、随访、因果对比与混杂控制由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。除非独立的识别与就绪门禁通过，结果保持仅分析级。我的研究问题是：",
        not_for=(
            ('The aim is to predict individual risk', '目的是预测个体风险', 'clinical-risk-prediction'),
            ('Treatment timing or confounding information is insufficient for causal interpretation', '治疗时间或混杂信息不足以支持因果解释', None),
        ),
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
            "exposure-outcome-distribution",
        ),
    ),
    MethodSkillSpec(
        "cross-sectional-phenotyping",
        "Cross-sectional phenotyping",
        "横断面表型聚类",
        "Phenotyping",
        "表型与亚型",
        "Group patients into subtypes from early ICU features and check how stable the grouping is; outcomes are not used to form the groups.",
        "用入 ICU 早期的特征把患者分成若干亚型，并检查分型是否稳定；结局不参与分型。",
        "phenotyping_cluster_v1",
        ("phenotyping.cluster_solution", "phenotyping.k_selection", "phenotyping.cluster_stability", "phenotyping.cluster_sizes", "phenotyping.outcome_by_cluster"),
        "deterministic_host",
        "reportable",
        "Use the Cross-sectional phenotyping method. The research plan proposes the feature window, outcome-free feature list, candidate k grid, and selection and stability rules; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“横断面表型聚类”方法。特征时间窗、不含结局的特征清单、候选 k 网格、选择与稳定性规则由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        not_for=(
            ('The shape of change over time defines the groups', '需要用随时间的变化形态定义分组', 'trajectory-phenotyping'),
            ('The aim is to predict a clinical outcome', '目的是预测临床结局', 'clinical-risk-prediction'),
        ),
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
        ),
    ),
    MethodSkillSpec(
        "trajectory-phenotyping",
        "Trajectory phenotyping and early subtype assignment",
        "轨迹表型发现与早期识别",
        "Phenotyping",
        "表型与亚型",
        "Group patients by how a measurement evolves over a fixed window, then explore whether earlier features can tell which group a new patient belongs to.",
        "按某项指标在固定时间窗内的变化轨迹给患者分型，再探索更早的临床特征能否判断新患者属于哪一型。",
        "trajectory_signed_phenotyping_v1",
        (
            "phenotyping.trajectory_feature_clustering",
            "phenotyping.trajectory_cluster_stability",
            "phenotyping.early_subtype_assignment",
        ),
        "deterministic_host",
        "analysis_only",
        "Use the Trajectory phenotyping and early subtype assignment method. The research plan proposes the trajectory window, representation, candidate grid, selection rules, and the early classifier's patient-level split; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“轨迹表型发现与早期识别”方法。轨迹时间窗、表示方法、候选网格、选择规则，以及早期分类器的患者级数据划分由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        not_for=(
            ('Only a single early feature summary is available', '只有一组早期特征汇总', 'cross-sectional-phenotyping'),
            ('The aim is to update future outcome risk', '目的是更新未来结局风险', 'dynamic-landmark-prediction'),
        ),
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
        ),
    ),
) + OWNER_SKILLS


_CAPABILITY_BY_ID = {row.capability_id: row for row in CAPABILITY_REGISTRY}
_ACTION_IDS = {row.action_id for row in HIGH_FREQUENCY_METHOD_ADAPTERS}

_FAMILY_ZH = {
    "data_preparation": "数据预处理",
    "time_to_event": "生存分析",
    "causal_emulation": "因果推断",
    "association": "关联分析",
    "prediction": "预测模型",
    "phenotyping": "表型与亚型",
    "descriptive": "描述性研究",
}

_COMPONENT_ZH = {
    "association.mixed_effects": ("重复测量线性混合模型", "用随机截距及可选时间随机斜率估计连续结局变化，报告区间、方差及 ICC。"),
    "association.gee": ("重复测量广义估计方程", "按预设工作相关结构估计二元或连续结局的平均变化，报告稳健标准误及可用的 QIC。"),
    "phenotyping.mixed_effects_growth": ("混合效应增长曲线", "用随机截距和时间随机斜率描述增长曲线，保留死亡脱落局限；不生成潜在类别。"),
    "data_preparation.outlier_rules": ("合理范围与异常值处理", "按变量选择范围过滤或固定边界截尾，报告移除或截尾数量；无原始数值的数据源不可执行。"),
    "data_preparation.window_aggregation": ("时间窗聚合", "逐变量审阅时间窗与首值、最小值、最大值、均值或末值；需要原始测量时间。"),
    "data_preparation.unit_harmonisation": ("单位统一状态", "只读查看概念字典在各数据库应用的单位换算规则，不重复换算。"),
    "data_preparation.missing_data": ("缺失数据处理与多重插补", "审阅完整病例或多重插补主分析及互补敏感性分析；记录随机种子并合并估计与不确定性。"),
    "data_preparation.scaling_coding": ("标准化与变量编码", "审阅标准化、参照水平及协变量样条结点；预测模型在训练分区内拟合预处理。"),
    "time_to_event.cox_hr": ("Cox 比例风险模型", "估计暴露与事件风险的调整后风险比。"),
    "time_to_event.km_logrank": ("Kaplan–Meier 与 log-rank", "展示分组生存曲线并进行未调整的 log-rank 比较。"),
    "time_to_event.ph_check": ("比例风险假设检验", "使用 Schoenfeld 残差检查 Cox 模型的比例风险假设。"),
    "time_to_event.subgroup_hr": ("亚组风险比与交互作用", "评估预设亚组中的效应异质性和交互作用。"),
    "time_to_event.rmst": ("限制性平均生存时间", "比较指定时间范围内两组的平均无事件时间。"),
    "time_to_event.competing_risks_cif": ("竞争风险累积发生率与 Gray 比较", "使用 Aalen–Johansen 方法描述目标事件的累积发生率，并进行分析级组间比较。"),
    "causal_emulation.iptw_or": ("稳定化 IPTW 因果对比", "在目标试验方案下估计边际因果对比。"),
    "causal_emulation.covariate_balance": ("协变量平衡与 Love plot", "用加权前后的标准化差异检查已测混杂平衡。"),
    "causal_emulation.positivity_overlap": ("阳性与重叠诊断", "检查治疗组之间的倾向评分重叠和极端权重。"),
    "causal_emulation.propensity_adjustment": ("倾向评分匹配与稳定化加权", "生成匹配或加权诊断，并检查重叠与协变量平衡。"),
    "causal_emulation.evalue": ("E-value 未测混杂敏感性", "量化未测混杂需要多强才能解释观察到的关联。"),
    "causal_emulation.negative_control": ("阴性对照分析", "使用预设的阴性对照结局或暴露检查残余偏倚。"),
    "causal_emulation.doubly_robust": ("双重稳健 AIPTW 敏感性", "在点干预、二分类结局条件下估计双重稳健平均处理效应。"),
    "causal_emulation.mediation": ("乘积法中介敏感性分析", "在明确识别假设下分解直接、间接和总效应。"),
    "association.ordinal_trend": ("有序趋势分析", "评估预设有序分组之间的剂量反应与单调趋势。"),
    "association.adjusted_association": ("调整后关联模型", "拟合参照组、变量编码和调整集都已写明的关联模型。"),
    "association.proportional_odds": ("有序结局比例优势模型", "按预设结局顺序估计共同优势比并报告比例优势检验。"),
    "association.free_days": ("存活且无需支持天数", "明确观察期限、支持概念与死亡赋分，报告推导审计和有序结局关联。"),
    "association.count_outcome": ("计数结局与发生率比", "按预设泊松或负二项模型处理暴露时间与过度离散，并报告发生率比。"),
    "association.multiple_adjustment": ("多层调整模型", "比较粗模型、最小调整和完整调整方案。"),
    "association.effect_modification": ("效应修饰与交互作用", "检查预设修饰因素对应的交互项与亚组结果。"),
    "association.missingness_audit": ("缺失数据与完整案例敏感性", "比较缺失模式、完整案例与预先声明的处理策略。"),
    "association.multiple_testing": ("多重检验校正", "使用 FDR 或 Bonferroni 等预设规则控制多重比较。"),
    "association.evalue": ("E-value 关联敏感性", "评估未测混杂解释调整后关联所需的最低强度。"),
    "association.robustness_panel": ("稳健性规格面板", "汇总不同队列或模型规格下主要估计的稳定性。"),
    "association.rcs_spline": ("限制性立方样条剂量反应", "检查连续暴露的非线性关联，并给出带置信区间的曲线。"),
    "prediction.discrimination_calibration": ("区分度与校准", "联合评估 AUROC、校准曲线和预测概率的可用性。"),
    "prediction.calibration_metrics": ("校准斜率、截距与 Brier 分数", "定量评估总体校准、校准斜率和概率误差。"),
    "prediction.delong_ci": ("DeLong AUROC 区间与比较", "计算 AUROC 置信区间并比较同一病例上的相关模型。"),
    "prediction.benchmark_comparison": ("与现有评分比较", "在同一批验证住院上比较模型与题目点名的现有评分：总比较区分度；评分是同一结局的概率时，也比较校准。"),
    "prediction.decision_curve": ("决策曲线与净获益", "在临床阈值范围内比较模型、全治和全不治策略。"),
    "prediction.threshold_metrics": ("临床阈值性能", "报告指定阈值下的敏感度、特异度、PPV 和 NPV。"),
    "prediction.feature_attribution": ("特征归因", "使用 SHAP、置换重要性或模型系数解释预测贡献。"),
    "prediction.subgroup_fairness": ("亚组与公平性性能", "比较人口学亚组中的区分度与校准表现。"),
    "prediction.internal_validation": ("内部验证", "用 bootstrap 或交叉验证评估并校正开发集乐观偏倚。"),
    "prediction.external_validation": ("跨数据库外部验证", "在其他 ICU 数据库应用冻结模型，报告区分度、校准和病例构成差异；单独报告再校准。"),
    "prediction.conformal_intervals": ("保形预测集", "生成具有边际或分层覆盖保证的预测集合。"),
    "prediction.dynamic_prediction": ("动态 Landmark 预测", "用每个预测时点之前的测量更新未来风险。"),
    "prediction.reclassification": ("NRI 与 IDI 重分类分析", "比较基线模型与更新模型在相同病例上的风险重分类。"),
    "phenotyping.cluster_solution": ("横断面亚表型聚类", "基于早期特征形成并检查患者亚表型。"),
    "phenotyping.k_selection": ("聚类数量选择", "使用 silhouette、gap 或 BIC 等预设标准选择聚类数。"),
    "phenotyping.cluster_stability": ("聚类稳定性与复现性", "通过重采样、共识或调整 Rand 指数检查稳定性。"),
    "phenotyping.trajectory_cluster_stability": ("轨迹聚类稳定性复拟合", "按预设方案重采样，复核所选轨迹模型的稳定性。"),
    "phenotyping.cluster_sizes": ("聚类规模与退化检查", "报告各聚类规模并标记近空或退化聚类。"),
    "phenotyping.outcome_by_cluster": ("聚类间结局描述", "描述已确定的各聚类之间的临床结局差异，不作因果解释。"),
    "phenotyping.trajectory_feature_clustering": ("轨迹特征聚类", "根据预设时间窗和表示方法构建纵向轨迹表型。"),
    "phenotyping.early_subtype_assignment": ("用早期特征识别轨迹亚型", "用轨迹观察窗之前的临床特征判断患者属于哪一个已确定的亚型，并在另一批患者上验证。"),
    "descriptive.descriptive_summary": ("队列与测量过程描述", "描述研究队列、变量分布和测量过程。"),
    "descriptive.table_one": ("基线特征表（Table 1）", "按组汇总参与者基线特征并保留真实分母。"),
    "descriptive.missingness_audit": ("缺失与完整性审计", "逐变量报告缺失、覆盖和测量可用性。"),
}


_COMPONENT_ZH["causal_emulation.g_methods"] = (
    "纵向边际结构模型与单时点 g 公式",
    "用序贯稳定 IPTW、可选删失权重和加权合并 logistic 拟合纵向 MSM，逐区间报告重叠、权重分布、有效样本量和截断；单时点 g 公式报告标准化风险差、风险比及 bootstrap 区间。尚不支持多区间 g 公式。",
)


def _method_component_catalog(*, enabled: bool) -> tuple[list[dict[str, Any]], int]:
    items: list[dict[str, Any]] = []
    planned_count = 0
    for suite in METHOD_SUITE_REGISTRY:
        for method in suite.methods:
            if method.implementation == "planned":
                planned_count += 1
                continue
            component_id = f"{suite.family}.{method.key}"
            title_zh, purpose_zh = _COMPONENT_ZH[component_id]
            execution_mode: ExecutionMode = (
                "deterministic_host"
                if method.implementation == "deterministic"
                else "agent_coded_with_host_gates"
            )
            items.append(
                {
                    "id": component_id,
                    "title": method.name,
                    "title_zh": title_zh,
                    "category": suite.label,
                    "category_zh": _FAMILY_ZH[suite.family],
                    "description": method.purpose,
                    "description_zh": purpose_zh,
                    "version": METHOD_SKILL_REGISTRY_VERSION,
                    "enabled": bool(enabled),
                    "method_family": suite.family,
                    "method_key": method.key,
                    "tier": method.tier,
                    "implementation": method.implementation,
                    "execution_mode": execution_mode,
                    "claim_ceiling": "analysis_only",
                    "outputs": [method.produces],
                    "reporting_items": list(method.reporting_items),
                    "kernel_modules": list(method.kernel_modules),
                    "prompt": (
                        f"Use {method.name} as a method component in the research plan. "
                        "The plan proposes its applicability, inputs, assumptions, and comparison for one review; "
                        "it stays analysis-only unless the enclosing reviewed workflow grants a higher ceiling. "
                        "My research question: "
                    ),
                    "prompt_zh": (
                        f"请在研究方案中使用“{title_zh}”方法组件。适用性、输入、假设和比较方式由研究计划提出，我在计划里一次审阅；"
                        "除非外层经过审阅的研究工作流明确授予更高权限，否则保持“仅分析”。我的研究问题是："
                    ),
                }
            )
    items.append({
        "id": "cross_database.meta_analysis", "title": "Per-database meta-analysis",
        "title_zh": "跨数据库效应合并", "category": "Evidence synthesis", "category_zh": "证据合成",
        "description": "Inverse-variance fixed effect and REML random effects yield a pooled observational estimate from independent, gate-passing databases with the same estimand. Changed estimands are excluded; execution approval does not authorize pooling them.",
        "description_zh": "对通过自身审核且估计目标一致的独立数据库进行固定效应和 REML 随机效应合并，结果仅为观察性关联。估计目标改变的数据库予以排除；运行批准不等于合并批准。",
        "version": METHOD_SKILL_REGISTRY_VERSION, "enabled": bool(enabled),
        "method_family": "cross_database", "method_key": "meta_analysis", "tier": "standard_supporting",
        "implementation": "deterministic", "execution_mode": "deterministic_host", "claim_ceiling": "analysis_only",
        "outputs": ["forest.svg", "heterogeneity table", "leave-one-database-out sensitivity", "exclusion reasons"],
        "reporting_items": ["Q with df and p", "I²", "tau²", "prediction interval", "exclusions", "leave-one-database-out",
                            "inverse-variance weights", "REML", "Hartung-Knapp option", "DerSimonian-Laird labelled sensitivity"],
        "kernel_modules": ["easyicu.research_agent.methods.sensitivity"],
        "prompt": "Pool verified per-database estimates with their coefficient SEs after equal-estimand and capability checks. Report REML, heterogeneity, exclusions and sensitivity; claim only a pooled observational estimate. My research question: ",
        "prompt_zh": "核对估计目标与数据库能力后，使用已核验的各数据库效应及系数标准误进行合并。报告 REML、异质性、排除原因和敏感性分析；仅描述合并后的观察性关联。我的研究问题是：",
    })
    return items, planned_count


def _validate_method_skills() -> None:
    ids = [row.skill_id for row in METHOD_SKILLS]
    if len(ids) != len(set(ids)):
        raise RuntimeError("method Skill ids must be unique")
    by_id = {row.skill_id: row for row in METHOD_SKILLS}
    registered_actions = {
        f"{suite.family}.{method.key}"
        for suite in METHOD_SUITE_REGISTRY
        for method in suite.methods
        if method.implementation != "planned"
    }
    for row in METHOD_SKILLS:
        if isinstance(row, OwnerSkillSpec):
            validate_owner_skill(row)
            continue
        if row.layer not in {"research_workflow", "analysis_module"}:
            raise RuntimeError("skill.analysis_layer_invalid")
        if row.layer == "research_workflow" and not row.not_for:
            raise RuntimeError("skill.not_for_required")
        for en, zh, sibling in row.not_for:
            if not en.strip() or not zh.strip():
                raise RuntimeError("skill.not_for_bilingual_required")
            if sibling is not None and (sibling not in by_id or sibling == row.skill_id):
                raise RuntimeError("skill.sibling_invalid")
        capability = _CAPABILITY_BY_ID.get(row.capability_id)
        if capability is None:
            raise RuntimeError(f"unknown method Skill capability: {row.capability_id}")
        unknown_actions = set(row.action_ids) - (_ACTION_IDS | registered_actions)
        if unknown_actions:
            raise RuntimeError(
                f"unknown method Skill actions for {row.skill_id}: {sorted(unknown_actions)}"
            )
        if row.claim_ceiling == "reportable" and (
            capability.scientific_validation != "reportable"
        ):
            raise RuntimeError(
                f"method Skill cannot exceed capability ceiling: {row.skill_id}"
            )
        unknown_modules = set(row.included_module_ids) - set(by_id)
        invalid_modules = {
            module_id
            for module_id in row.included_module_ids
            if module_id in by_id and by_id[module_id].layer != "analysis_module"
        }
        if unknown_modules or invalid_modules or row.skill_id in row.included_module_ids:
            raise RuntimeError(
                "method Skill composition drift for "
                f"{row.skill_id}: unknown={sorted(unknown_modules)!r}, "
                f"not_modules={sorted(invalid_modules)!r}, "
                f"self_reference={row.skill_id in row.included_module_ids}"
            )


_validate_method_skills()


def method_skill_catalog(*, enabled: bool) -> dict[str, Any]:
    """Return the public, digest-bound method catalogue for the desktop UI."""

    items = [row.to_dict(enabled=enabled) for row in METHOD_SKILLS]
    components, planned_count = _method_component_catalog(enabled=enabled)
    canonical = json.dumps(
        {"items": items, "components": components},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    workflows = [row for row in items if row["layer"] == "research_workflow"]
    modules = [row for row in items if row["layer"] == "analysis_module"]
    return {
        "version": METHOD_SKILL_REGISTRY_VERSION,
        "enabled": bool(enabled),
        "status": "enabled" if enabled else "disabled",
        "catalog_sha256": hashlib.sha256(canonical).hexdigest(),
        "items": items,
        "components": components,
        "workflow_count": len(workflows),
        "analysis_module_count": len(modules),
        "data_definition_count": sum(row["layer"] == "data_definition" for row in items),
        "planning_aid_count": sum(row["layer"] == "planning_aid" for row in items),
        "builtin_skill_count": len(items),
        "available_method_count": len(components),
        "planned_method_count": planned_count,
        "active_skill_ids": [row["id"] for row in items if row["enabled"]],
        "active_component_ids": [
            row["id"] for row in components if row["enabled"]
        ],
    }


class MethodSkillPackageNotFound(KeyError):
    """A requested built-in workflow or method component is not registered."""


def _package_entry(
    skill_id: str, *, enabled: bool
) -> tuple[str, dict[str, Any], str]:
    clean_id = str(skill_id or "").strip()
    catalog = method_skill_catalog(enabled=enabled)
    for kind, collection in (
        ("skill", catalog["items"]),
        ("method_component", catalog["components"]),
    ):
        for row in collection:
            if row["id"] == clean_id:
                resolved_kind = row.get("layer", kind)
                return resolved_kind, row, str(catalog["catalog_sha256"])
    raise MethodSkillPackageNotFound(clean_id)


# Reader content is typed separately from execution policy. The registries below
# remain the owners of actions, diagnostics, reasons and database dispositions.
Bilingual = tuple[str, str]


@dataclass(frozen=True, slots=True)
class WorkflowGuide:
    philosophy: Bilingual
    decisions: Bilingual
    question: Bilingual
    example_plan: Bilingual
    requirements: StudyRequirements
    output_files: tuple[str, ...]
    planned_filenames: bool = False
    demo_source: Literal["miiv_demo", "eicu_demo"] = "miiv_demo"


def _example_requirements(family: str, *, outcome: str | None = "hospital_death",
                          concepts: tuple[str, ...] = ("age", "sex", "lact"),
                          treatment: tuple[str, ...] = ()) -> StudyRequirements:
    return StudyRequirements(
        family=family, outcome=outcome, concepts=concepts, treatment_concepts=treatment,
        thresholds=(ThresholdRequirement(concept="age", operator=">=", value=18),),
        censoring_unit="hospitalization" if family == "survival" else None,
    )


WORKFLOW_GUIDES = {
    "repeated-measure-trajectory": WorkflowGuide(
        ("Estimate observed change on an explicit ICU time axis; death can cause informative dropout.",
         "在明确的 ICU 时间轴上估计实际观察到的变化；死亡可能导致信息性脱落。"),
        ("time origin/unit/window, bin width, aggregation, linear or spline time, random effects or GEE correlation, death-dropout sensitivity",
         "时间原点、单位、窗口、网格宽度、聚合方式、线性或样条时间、随机效应或 GEE 相关结构、死亡脱落敏感性分析"),
        ("How does daily lactate change during ICU days 0–6?", "ICU 第 0 至 6 天每日乳酸如何变化？"),
        ("Use the bound long panel; average measurements within each day from ICU admission. Fit a random-intercept/time-slope model; report CIs, covariance and ICC at time zero. State informative death dropout and a joint-model sensitivity note.",
         "使用已绑定的长表，从 ICU 入院起按日取均值。拟合随机截距及时间随机斜率模型，报告置信区间、协方差及零时点 ICC，明确死亡所致信息性脱落，并提出联合模型敏感性分析说明。"),
        _example_requirements("repeated_measures", outcome=None, concepts=("lact",)),
        ("repeated_measure_effects.csv", "repeated_measure_diagnostics.csv", "repeated_measure_profile.csv"),
    ),
    "adjusted-exposure-outcome-study": WorkflowGuide(
        ("Start with a clinical comparison and describe who was observed. Keep crude risks separate from the adjusted estimate. Adjustment addresses measured differences but does not establish a treatment effect.",
         "先明确临床比较，再描述实际观察到的人群。分别呈现各组粗率和调整后的估计。调整用于处理已测量的差异，不能据此认定治疗的因果效应。"),
        ("cohort and exclusions, time zero, exposure coding and reference, outcome horizon, covariates and missing-data handling",
         "队列与排除条件、时间零点、暴露编码与参照、结局观察期限、协变量和缺失数据处理"),
        ("Is lactate at ICU admission associated with death before hospital discharge?",
         "入 ICU 时的乳酸是否与出院前死亡相关？"),
        ("Adults; first ICU stay; ICU admission as time zero; first lactate at admission as a continuous exposure; hospital death as the outcome; age and sex adjustment; complete-case model with missingness reported; an odds ratio per mmol/L. The host must confirm that the lactate was available at time zero; later measurements require a revised landmark plan.",
         "成人、首次 ICU 住院；以入 ICU 为时间零点，入 ICU 时首个乳酸值作为连续暴露，院内死亡为结局；调整年龄和性别，报告缺失并使用完整案例，按每 mmol/L 报告比值比。系统须确认乳酸在时间零点已可获得；若只有之后的测量，计划需改用固定时间点设计。"),
        _example_requirements("association"),
        ("adjusted_association_estimates.csv", "adjusted_association_coefficients.csv"),
    ),
    "ordinal-dose-response-study": WorkflowGuide(
        ("Use an ordering that has clinical meaning before seeing outcomes. Show the population and absolute risks at every level. A trend estimate complements the level comparisons; sparse groups can make either unreliable.",
         "在查看结局前确定有临床意义的顺序。展示每个等级的人数与绝对风险。趋势估计补充各等级比较；人数稀少的组可能使两者都不可靠。"),
        ("ordered levels, reference and primary comparison, time zero, covariates, missingness and the trend model",
         "有序等级、参照及主要比较、时间零点、协变量、缺失处理和趋势模型"),
        ("Does death become more frequent across low, intermediate and high admission lactate groups?",
         "入 ICU 时乳酸从低、中到高分组，死亡是否更常见？"),
        ("Adults at ICU admission; first lactate available at admission, with levels below 2, 2 to below 4, and at least 4 mmol/L; lowest group as reference; hospital death; age and sex adjustment; complete-case analysis and a prespecified ordered trend. These cut points are example plan choices for review, not fitted results or a clinical recommendation.",
         "成人入 ICU 人群；使用入 ICU 时已可获得的首个乳酸，分为低于 2、2 至低于 4、至少 4 mmol/L，以最低组为参照；结局为院内死亡，调整年龄和性别；完整案例分析及预设有序趋势。这些界值是示例计划中供审阅的选择，并非拟合结果或临床建议。"),
        _example_requirements("association"), ("dose_response.csv",), True,
    ),
    "fixed-landmark-association-study": WorkflowGuide(
        ("Start the comparison among patients alive and still observed at the same landmark. Use only measurements obtained before that point to classify exposure. Report unknown exposure separately, and interpret the estimate only for patients eligible at the landmark.",
         "在同一固定时间点仍存活且仍在观察中的患者之间开始比较。只用此前获得的测量确定暴露分组。暴露不明者单独报告，估计仅适用于该时间点仍符合条件的人群。"),
        ("time zero, landmark and eligibility, exposure window and levels, reference group, outcome horizon, covariates and repeated stays",
         "时间零点、固定时间点及纳入条件、暴露时间窗与分组、参照组、结局观察期限、协变量及重复住院处理"),
        ("Among patients still in hospital after the first ICU day, is early lactate group associated with later hospital death?",
         "入 ICU 满一天仍在院的患者中，早期乳酸分组是否与此后的院内死亡相关？"),
        ("Adults; ICU admission as time zero; a 24-hour landmark; highest lactate in hours 0–24 grouped below 2, 2 to below 4, and at least 4 mmol/L; lowest group as reference; death after the landmark before hospital discharge; age and sex adjustment; patient-clustered uncertainty for repeated stays; unknown exposure described and excluded from the model.",
         "成人；以入 ICU 为时间零点，24 小时为固定时间点；按前 24 小时最高乳酸分为低于 2、2 至低于 4、至少 4 mmol/L，以最低组为参照；结局为该时间点后至出院的死亡；调整年龄和性别，重复住院按患者计算聚类稳健区间；描述暴露不明者并将其排除于模型之外。"),
        _example_requirements("association"),
        ("cohort_flow.csv", "measurement_audit.csv", "absolute_risk.csv",
         "adjusted_association_estimates.csv", "key_metrics.csv", "caveat_flags.json", "report.md", "manifest.json"),
    ),
    "time-varying-exposure-survival-study": WorkflowGuide(
        ("Represent exposure as it was known at each follow-up interval. Keep early events and unmeasured states visible instead of assigning future information to earlier time. The model describes an association and does not resolve time-dependent confounding.",
         "按每个随访区间当时已知的信息表示暴露。保留早期事件和未测量状态，不能把之后的信息填回之前。模型描述关联，不能解决随时间变化的混杂。"),
        ("time zero, interval boundaries, exposure updates, event and censoring definitions, horizon, baseline covariates and patient clustering",
         "时间零点、区间边界、暴露更新、事件与删失定义、观察期限、基线协变量及患者聚类"),
        ("As lactate changes during the stay, how is the most recent measurement associated with subsequent death?",
         "住院期间乳酸变化时，最近一次测量与之后的死亡有何关联？"),
        ("Adult ICU stays; ICU admission as time zero; split intervals at each lactate measurement; use only values already observed and preserve the unmeasured state; hospital death with discharge censoring; age and sex adjustment; patient-clustered uncertainty. Review the event-time definition before fitting.",
         "成人 ICU 住院；以入 ICU 为时间零点，在每次乳酸测量时分段，只使用已观察到的值并保留未测量状态；院内死亡为事件，出院时删失；调整年龄和性别，按患者计算聚类稳健区间。拟合前审阅事件时间定义。"),
        _example_requirements("survival"),
        ("time_varying_cox_estimates.csv", "time_varying_input_audit.csv", "time_varying_runtime_receipt.json"),
    ),
    "survival-time-to-event": WorkflowGuide(
        ("Compare time to an event using a shared start of follow-up. Patients contribute observed time until an event or the reviewed censoring rule. Kaplan–Meier curves describe survival; adjusted Cox results require review of the proportional-hazards diagnostic.",
         "用一致的随访起点比较到事件发生的时间。每位患者贡献观察时间，直到事件发生或达到审阅后的删失条件。Kaplan–Meier 曲线描述生存，解释调整后的 Cox 结果还需审阅比例风险检验。"),
        ("time origin, event and censoring rules, groups and reference, covariates, follow-up horizon and the proportional-hazards assessment",
         "时间起点、事件与删失规则、分组及参照、协变量、随访期限和比例风险评估"),
        ("Do adult ICU patients with higher admission lactate have a different time to hospital death?",
         "入 ICU 时乳酸较高的成人患者，到院内死亡的时间是否不同？"),
        ("First adult ICU stays; admission time origin; lactate at admission below 2 versus at least 2 mmol/L, with the lower group as reference; hospital death as event; censor at hospital discharge or day 28, whichever comes first; adjust for age and sex; review discharge censoring and proportional hazards.",
         "成人首次 ICU 住院；入 ICU 为起点，入 ICU 时乳酸低于 2 与至少 2 mmol/L 两组，以低值组为参照；院内死亡为事件，在出院或第 28 天中较早者删失；调整年龄和性别，审阅出院删失假设及比例风险检验。"),
        _example_requirements("survival"),
        ("survival_effect_estimates.csv", "survival_ph_diagnostic.csv", "survival_analysis_receipt.json"),
    ),
    "clinical-risk-prediction": WorkflowGuide(
        ("Define the clinical moment when a prediction would be made. Keep patients separate between development and validation and learn preprocessing only from development data. Assess both ranking and agreement between predicted and observed risk before discussing clinical use.",
         "先明确临床上何时作出预测。开发集和验证集按患者分开，预处理只从开发集学习。讨论临床用途前，同时评估风险排序和预测风险与实际风险的一致性。"),
        ("prediction time, outcome and horizon, predictors, patient split, preprocessing, internal validation and calibration",
         "预测时点、结局及期限、预测变量、患者划分、预处理、内部验证与校准"),
        ("Can information from the first ICU day predict later death before hospital discharge?",
         "ICU 第一天的信息能否预测之后出院前的死亡？"),
        ("Adults alive and in hospital at 24 hours; predict later hospital death from age, sex and lactate available by then; one logistic model; patient-disjoint development and validation sets; development-only imputation; report discrimination, calibration, Brier score and decision curves. Demo size may prevent a reliable fit.",
         "成人在 24 小时时仍存活且在院；用此前可获得的年龄、性别和乳酸预测之后的院内死亡；使用一个 logistic 模型，开发与验证患者不重叠，插补只在开发集学习；报告区分度、校准、Brier 分数和决策曲线。演示数据规模可能不足以可靠拟合。"),
        _example_requirements("prediction"),
        ("prediction_scores.csv", "prediction_performance.csv", "calibration_assessment.csv", "internal_validation.csv", "clinical_utility.csv"),
    ),
    "dynamic-landmark-prediction": WorkflowGuide(
        ("Reconsider risk only at prespecified prediction times. Separate the look-back window from the future outcome window at every time point. Performance is reported by landmark; good performance at one time does not establish performance at another.",
         "只在预先指定的时点更新风险。每个时点都将回看窗口与未来结局窗口分开。按预测时点报告性能，一个时点的良好表现不能证明另一个时点也可靠。"),
        ("landmarks, look-back windows, outcome horizons, predictors, patient split and observable follow-up",
         "预测时点、回看窗口、结局期限、预测变量、患者划分及可观察的随访"),
        ("Can we update the risk of death in the next ICU day as new measurements arrive?",
         "随着新测量到来，能否更新下一个 ICU 日内死亡的风险？"),
        ("Adults still in ICU at 24 and 48 hours; use age, sex and lactate from the preceding 24 hours to predict ICU death over the next 24 hours; keep all records from a patient in one partition; report follow-up observability and discrimination and calibration separately at each landmark.",
         "成人在 24 和 48 小时时仍在 ICU；使用年龄、性别及此前 24 小时乳酸预测之后 24 小时内 ICU 死亡；同一患者的所有记录放在同一数据集，逐时点报告随访是否可观察、区分度和校准。"),
        _example_requirements("dynamic_prediction", outcome="icu_death"),
        ("landmark_performance.csv", "landmark_observability.csv"), True,
    ),
    "target-trial-emulation": WorkflowGuide(
        ("Write down the hypothetical trial before analysing observational records. Align eligibility, treatment strategies and follow-up at the same time zero. Balance and overlap diagnostics can reveal problems but cannot prove that all confounding was measured.",
         "分析观察记录前先写清假想试验。使纳入条件、治疗策略和随访使用同一个时间零点。平衡与重叠诊断可以发现问题，但不能证明所有混杂都已被测量。"),
        ("eligibility, treatment strategies, time zero, follow-up, causal contrast, confounders, treatment timing and identification assumptions",
         "纳入条件、治疗策略、时间零点、随访、因果比较、混杂变量、治疗时间及因果识别假设"),
        ("Among adults needing blood-pressure support, how could we compare starting noradrenaline now with deferring it?",
         "对于需要血压支持的成人，如何比较现在开始去甲肾上腺素与推迟使用两种策略？"),
        ("Adults meeting the reviewed low-blood-pressure eligibility rule at ICU admission; compare initiation within one hour with no initiation in that hour; follow hospital death; propose adjustment using age, sex, admission lactate and blood pressure measured before treatment. The plan must assess treatment timing and confounding sufficiency; this teaching example does not establish causal identification.",
         "成人入 ICU 时满足审阅后的低血压条件；比较一小时内开始用药与该小时内不开始用药，随访院内死亡；计划提出用治疗前测得的年龄、性别、入院乳酸和血压调整。计划必须评估治疗时间与混杂信息是否充分；此教学示例不证明因果识别成立。"),
        _example_requirements("causal_inference", concepts=("age", "sex", "lact", "map", "norepi_rate"), treatment=("norepi_rate",)),
        ("trial_protocol.json", "causal_effect.csv", "covariate_balance.csv", "positivity_diagnostics.csv"), True,
    ),
    "cross-sectional-phenotyping": WorkflowGuide(
        ("Group patients using early clinical features without including outcomes in the clustering. Compare the prespecified candidate cluster counts and examine stability. Describe outcomes after groups are defined; different outcomes do not prove biological subtypes.",
         "用早期临床特征给患者分组，聚类时不加入结局。比较预设的候选聚类数并检查稳定性。分组确定后再描述结局；结局不同不能证明存在生物学亚型。"),
        ("feature window, outcome-free feature list, scaling, missingness, candidate cluster counts, selection and stability rules",
         "特征时间窗、不含结局的特征清单、尺度处理、缺失处理、候选聚类数、选择与稳定性规则"),
        ("Do adults have reproducible early ICU profiles based on lactate, heart rate and blood pressure?",
         "能否根据乳酸、心率和血压找到可复现的成人早期 ICU 特征分组？"),
        ("First adult ICU stays; summarize lactate, heart rate and mean arterial pressure over hours 0–6; standardize features without outcomes; compare two to four clusters with the reviewed selection and resampling rules; describe cluster sizes and hospital outcomes only after selecting groups.",
         "成人首次 ICU 住院；汇总前 6 小时乳酸、心率和平均动脉压，不含结局地进行特征标准化；按审阅后的选择和重采样规则比较二至四组的方案，确定分组后再描述组大小与院内结局。"),
        _example_requirements("phenotyping", concepts=("age", "lact", "hr", "map")),
        ("phenotype_profiles.csv", "phenotype_assignments.csv", "cluster_selection.csv", "cluster_stability_with_algorithm_agreement.csv"),
    ),
    "trajectory-phenotyping": WorkflowGuide(
        ("Discover patterns only within the reviewed observation window. Fix the selected groups and label meanings before training an early classifier. Test early assignment using earlier features and patient-disjoint development and validation sets; assignment accuracy does not validate the biological meaning of a group.",
         "只在审阅后的观察窗口内发现变化模式。训练早期分类器前先确定所选分组和标签含义。用更早的特征和互不重叠的开发、验证患者评估早期识别；识别准确率不能证明分组具有生物学意义。"),
        ("trajectory window and representation, missingness, candidate models, selection and stability rules, early feature window and patient split",
         "轨迹窗口与表示方法、缺失处理、候选模型、选择与稳定性规则、早期特征窗口及患者划分"),
        ("Are there stable blood-pressure trajectories after ICU admission, and can earlier features identify these groups?",
         "入 ICU 后是否存在稳定的血压轨迹分组，更早的特征能否识别这些组？"),
        ("Adults with the reviewed observation opportunity; mean arterial pressure trajectories in hours 6–24; a prespecified candidate model grid and resampling assessment; age and lactate from hours 0–6 for early assignment; patient-disjoint development and validation; report discovery and assignment separately.",
         "具有审阅后要求的观察机会的成人；使用入 ICU 后 6 至 24 小时平均动脉压轨迹，预设候选模型网格并做重采样评估；以此前 0 至 6 小时的年龄和乳酸信息进行早期识别，开发与验证患者不重叠；分别报告发现分组和早期识别结果。"),
        _example_requirements("trajectory_clustering", outcome=None, concepts=("age", "lact", "map")),
        ("trajectory_candidate_selection.csv", "candidate_cluster_assignments.csv", "cluster_selection.json"),
    ),
}


@dataclass(frozen=True, slots=True)
class MethodNotes:
    """What a workflow's estimators are, who computes them, and how to read them.

    Each line restates the owner's implementation; the executors and kernels
    remain the authority for what is computed.
    """

    execution: Bilingual
    estimators: tuple[Bilingual, ...]
    interpretation: tuple[Bilingual, ...]
    literature: tuple[str, ...]


_OR_COMMON_OUTCOME = (
    "An odds ratio overstates the risk ratio when the outcome is common (roughly above 10 %, as ICU mortality often is); read it with the absolute risks and do not report it as a relative risk.",
    "结局常见时（大约高于 10%，ICU 死亡率常如此），比值比会夸大风险比；应结合绝对风险解读，不能当作相对风险报告。",
)
_EVENTS_PER_PARAMETER = (
    "Fewer than about 10 outcome events per model parameter makes the estimates unstable; the plan counts parameters, not variables.",
    "每个模型参数对应的结局事件少于约 10 个时，估计不稳定；按模型参数计数，而非按变量个数。",
)
_STROBE_RECORD = (
    "von Elm E, et al. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement. Lancet. 2007;370:1453-1457.",
    "Benchimol EI, et al. The REporting of studies Conducted using Observational Routinely-collected health Data (RECORD) statement. PLoS Med. 2015;12:e1001885.",
)
_CLUSTER_SE = "Liang KY, Zeger SL. Longitudinal data analysis using generalized linear models. Biometrika. 1986;73:13-22."
_TRIPOD = (
    "Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD). BMJ. 2015;350:g7594.",
    "Collins GS, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378.",
)

METHOD_NOTES = {
    "repeated-measure-trajectory": MethodNotes(
        ("A reviewed host owner constructs the long panel and calls statsmodels; artifacts contain aggregate results only.",
         "经审阅的主机执行组件构建长表并调用 statsmodels，产物只包含汇总结果。"),
        (("LMM: correlated random intercept and optional time slope, fixed-effect CIs and covariance; ICC refers to time zero when slopes vary.",
          "LMM：相关随机截距及可选时间斜率，报告固定效应区间和协方差；斜率随机时 ICC 指零时点。"),
         ("GEE: binary/continuous outcome with robust SE and prespecified exchangeable, AR1 or independent working correlation; QIC when available.",
          "GEE：二元或连续结局，使用稳健标准误和预设交换型、AR1 或独立工作相关结构，报告可用的 QIC。")),
        (("Death truncates measurements: survivor restriction changes the estimand and does not correct informative dropout. A joint model is a note, not a fitted result.",
          "死亡截断测量：幸存者限制改变估计目标，不能校正信息性脱落。联合模型仅为后续分析说明，并未实际拟合。"),
         ("Growth curves have no latent classes. LCGA/GBTM remains planned until the runner has a validated package and BIC, APPA, entropy and group-size diagnostics.",
          "增长曲线不生成潜在类别。LCGA/GBTM 须待运行镜像提供已验证软件包及 BIC、APPA、熵和组大小诊断后开放。")),
        ("Laird NM, Ware JH. Random-effects models for longitudinal data. Biometrics. 1982;38:963–974.",
         "Liang KY, Zeger SL. Longitudinal data analysis using generalized linear models. Biometrika. 1986;73:13–22."),
    ),
    "adjusted-exposure-outcome-study": MethodNotes(
        ("EasyICU computes the primary model itself; the AI does not write analysis code for it.",
         "主要模型由 EasyICU 直接计算，不由 AI 编写分析代码。"),
        (("Binary outcome: logistic regression (statsmodels maximum likelihood), reported as an odds ratio with a 95 % confidence interval. Continuous outcome: linear regression (ordinary least squares), reported as a mean difference.",
          "二分类结局：logistic 回归（statsmodels 最大似然估计），报告比值比及 95% 置信区间。连续结局：线性回归（最小二乘），报告均值差。"),
         ("The coding, levels, reference and transform of every model term come from the reviewed plan. A rank-deficient design or a non-binary outcome for a logistic model stops the run; no term is dropped silently.",
          "每个模型项的编码、水平、参照和变换都来自审阅后的计划。设计矩阵秩不足，或 logistic 模型的结局不是二分类时，运行停止，不会悄悄删除变量。"),
         ("When the plan declares that repeated ICU stays of one patient are dependent, standard errors are cluster-robust (sandwich) by patient; otherwise they are model-based.",
          "计划声明同一患者的多次 ICU 住院相互关联时，按患者计算聚类稳健（三明治）标准误；否则使用模型标准误。"),
         ("Missing data: complete-case fit; the number of rows used and dropped is recorded. The robustness panel reuses the same estimator, so primary and sensitivity estimates differ only by their declared change.",
          "缺失数据：完整案例拟合，记录纳入和剔除的行数。稳健性分析复用同一估计程序，主分析与敏感性分析只在声明的改动上不同。")),
        (_OR_COMMON_OUTCOME, _EVENTS_PER_PARAMETER,
         ("Complete-case estimates can be biased when missingness depends on the outcome; check the complete-case share before interpreting.",
          "缺失与结局有关时，完整案例估计可能有偏；解读前先查看完整案例所占比例。"),
         ("Adjustment addresses measured confounders only; the estimate is an association, not a treatment effect.",
          "调整只处理已测量的混杂；估计值是关联，不是治疗效应。")),
        (*_STROBE_RECORD, _CLUSTER_SE,
         "Peduzzi P, et al. A simulation study of the number of events per variable in logistic regression analysis. J Clin Epidemiol. 1996;49:1373-1379.",
         "Zhang J, Yu KF. What's the relative risk? A method of correcting the odds ratio in cohort studies of common outcomes. JAMA. 1998;280:1690-1691."),
    ),
    "ordinal-dose-response-study": MethodNotes(
        ("The AI writes the analysis code from the reviewed plan. EasyICU supplies validated trend tests for that code to call, checks the declared levels and the result tables, and keeps the result analysis only.",
         "分析代码由 AI 按审阅后的计划编写。EasyICU 提供经过验证的趋势检验供其调用，核对预设等级和结果表，结果保持仅分析级。"),
        (("Built-in unadjusted tests: Cochran–Armitage trend test for a binary outcome across ordered levels; Jonckheere–Terpstra test for an ordinal or continuous outcome; Wilson intervals for each level's absolute risk.",
          "内置的未调整检验：二分类结局用 Cochran–Armitage 趋势检验；有序或连续结局用 Jonckheere–Terpstra 检验；各等级绝对风险用 Wilson 区间。"),
         ("Adjusted trend: a regression with the level index as one term (trend per level) and, separately, the levels as indicators against the reference (level contrasts), as the plan declares.",
          "调整后的趋势：按计划在回归中以等级序号作为一个项（每升一级的趋势），并另以各等级相对参照的指示变量估计等级间对比。"),
         ("Non-linearity: the trend model is compared with the indicator model; sparse levels are reported, not merged after outcomes are seen.",
          "非线性：比较趋势模型与指示变量模型；人数稀少的等级如实报告，不在查看结局后合并。")),
        (("A significant trend does not mean that every step raises the risk; read the level-by-level absolute risks.",
          "趋势显著不代表每升一级风险都升高；应逐级查看绝对风险。"),
         ("Scoring the levels 1, 2, 3 assumes equal spacing on the model scale; the plan states the scores used.",
          "把等级记为 1、2、3 相当于假设各级在模型尺度上等距；计划须写明所用分值。"),
         _OR_COMMON_OUTCOME,
         ("Cut points must be fixed before outcomes are seen.", "分组界值必须在查看结局前确定。")),
        ("Cochran WG. Some methods for strengthening the common chi-squared tests. Biometrics. 1954;10:417-451.",
         "Armitage P. Tests for linear trends in proportions and frequencies. Biometrics. 1955;11:375-386.",
         "Jonckheere AR. A distribution-free k-sample test against ordered alternatives. Biometrika. 1954;41:133-145.",
         "Wilson EB. Probable inference, the law of succession, and statistical inference. J Am Stat Assoc. 1927;22:209-212.",
         *_STROBE_RECORD),
    ),
    "fixed-landmark-association-study": MethodNotes(
        ("EasyICU computes every estimate itself; this package's reference scripts call the same code.",
         "所有估计都由 EasyICU 直接计算；本包的参考脚本调用同一套代码。"),
        (("Eligibility at the landmark: alive and still observed at the landmark; exposure is classified only from measurements before it, and follow-up starts at the landmark.",
          "固定时间点的纳入条件：该时间点仍存活且仍在观察中；只用此前的测量确定暴露分组，随访从该时间点开始。"),
         ("Primary model: logistic regression with the declared contrasts and covariates, cluster-robust standard errors by patient when repeated stays are declared dependent.",
          "主要模型：按预设对比和协变量拟合 logistic 回归；声明重复住院相关时按患者计算聚类稳健标准误。"),
         ("Absolute risk per exposure level; unknown exposure is reported as its own state and excluded from the models. Ordered levels add an adjusted per-level trend and Cochran–Armitage / Jonckheere–Terpstra tests with Holm adjustment.",
          "报告各暴露等级的绝对风险；暴露不明者单列报告，不进入模型。有序等级另报告调整后的每级趋势，以及经 Holm 校正的 Cochran–Armitage / Jonckheere–Terpstra 检验。")),
        (("Estimates apply only to patients alive and observed at the landmark; changing the landmark changes the population.",
          "估计只适用于该时间点仍存活且在观察中的患者；改变时间点就改变了人群。"),
         _OR_COMMON_OUTCOME, _EVENTS_PER_PARAMETER,
         ("A large unknown-exposure share is itself a finding; it is flagged above 10 % by default.",
          "暴露不明比例较高本身就是需要报告的发现；默认超过 10% 时标记。")),
        ("Anderson JR, Cain KC, Gelber RD. Analysis of survival by tumor response. J Clin Oncol. 1983;1:710-719.",
         "Dafni U. Landmark analysis at the 25-year landmark point. Circ Cardiovasc Qual Outcomes. 2011;4:363-371.",
         _CLUSTER_SE, *_STROBE_RECORD),
    ),
    "time-varying-exposure-survival-study": MethodNotes(
        ("EasyICU computes the model itself with the R survival package; results stay analysis only.",
         "模型由 EasyICU 调用 R survival 包直接计算；结果保持仅分析级。"),
        (("Counting-process (start, stop] intervals split at each exposure update. Only values already observed are carried forward, and the unmeasured state is kept rather than filled.",
          "按计数过程把随访切成 (start, stop] 区间，在每次暴露更新时分段。只沿用已观察到的值，未测量状态予以保留而不填补。"),
         ("Cox model with a time-varying exposure: survival::coxph with Efron ties and a robust sandwich variance clustered by patient.",
          "含时变暴露的 Cox 模型：survival::coxph，Efron 法处理同时事件，按患者聚类的稳健（三明治）方差。"),
         ("Checks: early events retained, interval coverage, convergence, finite covariance and alternative update rules.",
          "检查：早期事件是否保留、区间覆盖、收敛、协方差有限，以及替代更新规则。")),
        (("Time-dependent confounding (a covariate affected by earlier exposure that also drives later exposure) is not resolved by this model; that requires methods such as marginal structural models.",
          "本模型不能处理随时间变化的混杂（受此前暴露影响、又影响之后暴露的协变量）；这需要边际结构模型等方法。"),
         ("The hazard ratio compares current exposure states; it is not a cumulative or causal effect.",
          "风险比比较的是当前暴露状态，不是累积效应或因果效应。"),
         ("Measurement frequency can be informative: sicker patients are measured more often.",
          "测量频率本身可能带有信息：病情越重，测得越频繁。")),
        ("Andersen PK, Gill RD. Cox's regression model for counting processes: a large sample study. Ann Stat. 1982;10:1100-1120.",
         "Therneau TM, Grambsch PM. Modeling Survival Data: Extending the Cox Model. Springer; 2000.",
         "Fisher LD, Lin DY. Time-dependent covariates in the Cox proportional-hazards regression model. Annu Rev Public Health. 1999;20:145-157.",
         "Lin DY, Wei LJ. The robust inference for the Cox proportional hazards model. J Am Stat Assoc. 1989;84:1074-1078.",
         "Robins JM, Hernán MA, Brumback B. Marginal structural models and causal inference in epidemiology. Epidemiology. 2000;11:550-560."),
    ),
    "survival-time-to-event": MethodNotes(
        ("EasyICU computes the primary model itself; the AI does not write analysis code for it.",
         "主要模型由 EasyICU 直接计算，不由 AI 编写分析代码。"),
        (("Kaplan–Meier survival curves by group.", "按组绘制 Kaplan–Meier 生存曲线。"),
         ("Cox proportional hazards regression (lifelines, Efron ties) for the adjusted hazard ratio of the declared exposure, with a 95 % confidence interval.",
          "Cox 比例风险回归（lifelines，Efron 法处理同时事件）估计预设暴露的调整后风险比及 95% 置信区间。"),
         ("Proportional-hazards check: Schoenfeld-residual test against Kaplan–Meier-transformed time for each model term, with a family-wise global row.",
          "比例风险检验：对每个模型项做 Schoenfeld 残差检验（时间按 Kaplan–Meier 变换），并给出族系水平的总体结果。")),
        (("If proportional hazards fail, a single hazard ratio averages a changing effect; consider restricted mean survival time or a time-split estimate.",
          "比例风险不成立时，单一风险比是对随时间变化效应的平均；可考虑限制性平均生存时间或分时段估计。"),
         ("Censoring at hospital discharge assumes discharged patients share the future risk of those who stay, which rarely holds; prefer a fixed horizon with known vital status or a competing-risk analysis, and state the choice in the plan.",
          "在出院时删失，相当于假设出院者此后的风险与仍住院者相同，这很少成立；更宜采用已知生存状态的固定时限，或竞争风险分析，并在计划中写明。"),
         _EVENTS_PER_PARAMETER,
         ("A hazard ratio is not a risk ratio; report absolute survival at the horizon as well.",
          "风险比不是风险之比；还应报告观察期限时的绝对生存率。")),
        ("Kaplan EL, Meier P. Nonparametric estimation from incomplete observations. J Am Stat Assoc. 1958;53:457-481.",
         "Cox DR. Regression models and life-tables. J R Stat Soc Series B. 1972;34:187-220.",
         "Efron B. The efficiency of Cox's likelihood function for censored data. J Am Stat Assoc. 1977;72:557-565.",
         "Grambsch PM, Therneau TM. Proportional hazards tests and diagnostics based on weighted residuals. Biometrika. 1994;81:515-526.",
         "Peduzzi P, et al. Importance of events per independent variable in proportional hazards regression analysis. II. Accuracy and precision of regression estimates. J Clin Epidemiol. 1995;48:1503-1510.",
         "Austin PC, Lee DS, Fine JP. Introduction to the analysis of survival data in the presence of competing risks. Circulation. 2016;133:601-609."),
    ),
    "clinical-risk-prediction": MethodNotes(
        ("EasyICU fits and evaluates the model itself; the AI does not write the model code.",
         "模型由 EasyICU 直接拟合和评估，不由 AI 编写模型代码。"),
        (("Logistic regression with an L2 (ridge) penalty, C = 1.0 fixed rather than tuned (scikit-learn).",
          "带 L2（岭）惩罚的 logistic 回归，C = 1.0 固定、不调参（scikit-learn）。"),
         ("Preprocessing learned on development patients only: median imputation and standardization for numeric predictors; most-frequent imputation and one-hot coding for categorical predictors.",
          "预处理只从开发集患者学习：数值预测变量用中位数插补并标准化；分类预测变量用众数插补并独热编码。"),
         ("Patient-level split: 80 % development and 20 % validation with a fixed seed, so no patient's stays appear in both.",
          "按患者划分：80% 开发、20% 验证，随机种子固定，同一患者的住院不会同时出现在两边。"),
         ("Validation: AUROC with a DeLong interval (or a patient-resampling percentile interval); calibration intercept, slope and curve; Brier score; decision-curve net benefit.",
          "验证：AUROC 及 DeLong 区间（或按患者重抽样的百分位区间）；校准截距、斜率和校准曲线；Brier 分数；决策曲线净获益。")),
        (("A single split uses data less efficiently than bootstrap optimism correction, especially in small samples; read the interval widths.",
          "单次划分对数据的利用不如 bootstrap 乐观校正充分，样本小时尤甚；应关注区间宽度。"),
         ("A calibration slope below 1 indicates overfitting even when the AUROC looks good.",
          "即使 AUROC 看起来不错，校准斜率小于 1 也提示过拟合。"),
         ("Check the number of outcome events against published sample-size criteria before trusting performance.",
          "相信模型性能前，先按已发表的样本量标准核对结局事件数。"),
         ("Internal validation does not show that the model transports to another hospital or database.",
          "内部验证不能说明模型可推广到其他医院或数据库。")),
        (*_TRIPOD,
         "DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. Biometrics. 1988;44:837-845.",
         "Van Calster B, et al. Calibration: the Achilles heel of predictive analytics. BMC Med. 2019;17:230.",
         "Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. Med Decis Making. 2006;26:565-574.",
         "Steyerberg EW, et al. Internal validation of predictive models: efficiency of some procedures for logistic regression analysis. J Clin Epidemiol. 2001;54:774-781.",
         "Riley RD, et al. Calculating the sample size required for developing a clinical prediction model. BMJ. 2020;368:m441."),
    ),
    "dynamic-landmark-prediction": MethodNotes(
        ("The AI writes the model code from the reviewed plan. EasyICU supplies validated functions that build the landmark rows, attach outcomes and evaluate predictions, checks the outputs, and keeps the result analysis only.",
         "模型代码由 AI 按审阅后的计划编写。EasyICU 提供经过验证的函数来构建各时点的数据行、连接结局并评估预测，核对输出，结果保持仅分析级。"),
        (("Stacked landmark data: one row per patient still at risk at each landmark, predictors from its look-back window, the outcome in the horizon after it.",
          "堆叠的 landmark 数据：每个时点仍处于风险中的患者各占一行，预测变量取自该时点的回看窗口，结局取该时点之后的期限内。"),
         ("The plan chooses the model, commonly a logistic model per landmark or one model with landmark terms.",
          "模型由计划选择，常见做法是每个时点一个 logistic 模型，或一个含时点项的模型。"),
         ("Evaluation at each landmark: discrimination, calibration and how much follow-up was observable.",
          "逐时点评估：区分度、校准，以及可观察随访的比例。")),
        (("Rows from one patient are correlated; keep each patient in one partition and use patient-clustered uncertainty.",
          "同一患者的多行彼此相关；每位患者只放在一个数据集中，不确定性按患者聚类计算。"),
         ("Risk is conditional on being alive and in the ICU at the landmark; performance at one landmark does not carry to another.",
          "风险以在该时点仍存活且在 ICU 为条件；一个时点的性能不能推及其他时点。")),
        ("van Houwelingen HC. Dynamic prediction by landmarking in event history analysis. Scand J Stat. 2007;34:70-85.",
         "van Houwelingen HC, Putter H. Dynamic Prediction in Clinical Survival Analysis. CRC Press; 2011.",
         _TRIPOD[1]),
    ),
    "target-trial-emulation": MethodNotes(
        ("Two routes. A grace-period comparison of starting a treatment within a window versus not starting it, with death by a fixed horizon, runs EasyICU's own clone-censor-weight estimator. Other designs are coded by the AI from the reviewed protocol, checked by EasyICU and kept analysis only.",
         "分两种情况。比较“在宽限期内开始某治疗”与“不开始”、结局为固定期限内死亡时，由 EasyICU 自己的克隆-删失-加权程序计算。其他设计由 AI 按审阅后的方案编写代码，经 EasyICU 核验，结果保持仅分析级。"),
        (("Clone-censor-weight: every eligible stay is cloned into both strategies, and a clone is censored when its stay deviates from its strategy.",
          "克隆-删失-加权：每个符合条件的住院复制到两种策略中，住院偏离所属策略时删失该副本。"),
         ("Inverse probability of censoring weights from pooled hourly logistic models (statsmodels GLM) for starting the treatment and for leaving the ICU, with stabilized numerators.",
          "删失的逆概率权重来自按小时汇总的 logistic 模型（statsmodels GLM），分别拟合开始治疗和离开 ICU 的概率，分子做稳定化处理。"),
         ("Weighted Kaplan–Meier risk of death by the horizon in each arm; risk difference and risk ratio with patient-bootstrap percentile intervals. Sensitivity analyses: weights clipped at the 1st and 99th percentiles, and unweighted.",
          "每组用加权 Kaplan–Meier 估计期限内死亡风险；报告风险差和风险比及按患者 bootstrap 的百分位区间。敏感性分析：权重截断于第 1 和第 99 百分位，以及不加权。"),
         ("The run stops on too few events, an unobserved strategy, a positivity violation, extreme weights or an unstable bootstrap.",
          "事件过少、某策略未被观察到、违反正值性、权重极端或 bootstrap 不稳定时，运行停止。")),
        (("Validity rests on no unmeasured confounding of treatment start and ICU exit, on positivity and on correct weight models; balance diagnostics cannot prove the first.",
          "结论成立依赖于：治疗开始和离开 ICU 不存在未测量混杂、满足正值性、权重模型正确；平衡诊断无法证明第一条。"),
         ("Aligning eligibility, assignment and time zero prevents immortal-time bias.",
          "让纳入、分组与时间零点对齐，可避免不朽时间偏倚。"),
         ("The estimand is per protocol (adherence to a strategy), not intention to treat.",
          "估计目标是按方案（坚持某一策略）的效应，不是意向性治疗效应。")),
        ("Hernán MA, Robins JM. Using big data to emulate a target trial when a randomized trial is not available. Am J Epidemiol. 2016;183:758-764.",
         "Hernán MA, Sauer BC, Hernández-Díaz S, Platt R, Shrier I. Specifying a target trial prevents immortal time bias and other self-inflicted injuries in observational analyses. J Clin Epidemiol. 2016;79:70-75.",
         "Cain LE, et al. When to start treatment? A systematic approach to the comparison of dynamic regimes using observational data. Int J Biostat. 2010;6:Article 18.",
         "Cole SR, Hernán MA. Constructing inverse probability weights for marginal structural models. Am J Epidemiol. 2008;168:656-664.",
         "Maringe C, et al. Reflection on modern methods: trial emulation in the presence of immortal-time bias. Int J Epidemiol. 2020;49:1719-1729."),
    ),
    "cross-sectional-phenotyping": MethodNotes(
        ("EasyICU computes the clustering itself; outcomes are never clustering inputs.",
         "聚类由 EasyICU 直接计算；结局从不作为聚类输入。"),
        (("Features from the reviewed early window, median-imputed and standardized.",
          "特征取自审阅后的早期时间窗，经中位数插补并标准化。"),
         ("k-means (scikit-learn MiniBatchKMeans, 10 starts, fixed seed) over the plan's candidate grid (default k = 2 to 6); k is chosen by the highest silhouette, ties going to the smaller k.",
          "k-means（scikit-learn MiniBatchKMeans，10 次初始化，固定种子），在计划的候选网格（默认 k = 2 至 6）中按最高轮廓系数选择 k，相同时取较小的 k。"),
         ("Stability: refits on 80 % subsamples compared with the primary assignment by adjusted Rand index; agreement with a diagonal Gaussian mixture model; a complete-case sensitivity with a bootstrap interval.",
          "稳定性：在 80% 子样本上重新拟合，用调整兰德指数与主分组比较；与对角协方差高斯混合模型比较一致性；另做完整案例敏感性分析并给出 bootstrap 区间。"),
         ("Outcomes are described by cluster only after the solution is chosen.",
          "确定分组后才按组描述结局。")),
        (("Clusters partition this sample; they are not proven biological entities, and replication in another database is the stronger test.",
          "聚类只是对本样本的划分，并非已证实的生物学实体；在另一个数据库中重现是更有力的检验。"),
         ("The silhouette favors compact clusters of similar size; a low value means weak structure.",
          "轮廓系数偏好紧凑、大小相近的簇；数值低说明结构不明显。"),
         ("Different outcomes across clusters do not validate the clusters.",
          "各组结局不同，不能证明分组有效。")),
        ("Seymour CW, et al. Derivation, validation, and potential treatment implications of novel clinical phenotypes for sepsis. JAMA. 2019;321:2003-2017.",
         "Calfee CS, et al. Subphenotypes in acute respiratory distress syndrome: latent class analysis of data from two randomised controlled trials. Lancet Respir Med. 2014;2:611-620.",
         "Rousseeuw PJ. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. J Comput Appl Math. 1987;20:53-65.",
         "Hubert L, Arabie P. Comparing partitions. J Classif. 1985;2:193-218.",
         "Hennig C. Cluster-wise assessment of cluster stability. Comput Stat Data Anal. 2007;52:258-271."),
    ),
    "trajectory-phenotyping": MethodNotes(
        ("Trajectory discovery runs EasyICU's own estimator. The early-assignment classifier is coded by the AI from the reviewed plan and is exploratory, so the workflow as a whole stays analysis only.",
         "轨迹分型由 EasyICU 自己的程序计算。早期识别分类器由 AI 按审阅后的计划编写，属探索性分析，因此整个流程的结果保持仅分析级。"),
        (("Fixed-window representation of the measurement, one coordinate per window summary.",
          "在固定时间窗内表示指标，每个窗口汇总值作为一个坐标。"),
         ("Mixed-mode latent class model fitted by EM: class-specific Gaussians for continuous coordinates and level probabilities for ordinal ones (such as organ scores); each row contributes its observed coordinates (full-information likelihood under missing at random).",
          "用 EM 拟合混合型潜类别模型：连续坐标为各类别的高斯分布，有序坐标（如器官评分）为各类别的水平概率；每行只贡献其已观察的坐标（随机缺失假设下的全信息似然）。"),
         ("The class count is chosen by BIC over the prespecified grid; an optimum on the grid boundary is reported as no solution. Stability is assessed by resampling (100 resamples by default).",
          "类别数在预设网格中按 BIC 选择；最优值落在网格边界时报告为无解。稳定性通过重抽样评估（默认 100 次）。"),
         ("Early assignment: a multinomial logistic classifier on features from before the window, with patient-disjoint development and validation sets, reporting balanced accuracy, macro-F1, log loss and multiclass Brier score.",
          "早期识别：用时间窗之前的特征拟合多分类 logistic 分类器，开发与验证患者不重叠，报告平衡准确率、宏 F1、对数损失和多分类 Brier 分数。")),
        (("BIC selection depends on the representation; very small classes may be artifacts.",
          "BIC 的选择取决于表示方法；很小的类别可能是伪影。"),
         ("Patients must be observed through the window, which conditions on surviving it.",
          "患者须在整个时间窗内被观察，这相当于以活过该时间窗为条件。"),
         ("Assignment accuracy does not establish that the groups are biologically meaningful.",
          "识别准确率不能说明分组具有生物学意义。")),
        ("Nagin DS. Analyzing developmental trajectories: a semiparametric, group-based approach. Psychol Methods. 1999;4:139-157.",
         "Dempster AP, Laird NM, Rubin DB. Maximum likelihood from incomplete data via the EM algorithm. J R Stat Soc Series B. 1977;39:1-38.",
         "Schwarz G. Estimating the dimension of a model. Ann Stat. 1978;6:461-464.",
         "Vermunt JK, Magidson J. Latent class cluster analysis. In: Hagenaars JA, McCutcheon AL, eds. Applied Latent Class Analysis. Cambridge University Press; 2002:89-106.",
         "Bhavani SV, et al. Identifying novel sepsis subphenotypes using temperature trajectories. Am J Respir Crit Care Med. 2019;200:327-335.",
         _TRIPOD[1]),
    ),
}


def _method_lines(row: dict[str, Any], language: int) -> list[str]:
    if row["id"] == "cross_database.meta_analysis":
        return [("Inverse-variance fixed effect and REML random effects. Optional unmodified Hartung-Knapp t intervals; DerSimonian-Laird is a labelled sensitivity. Ratios use log coefficients and their SEs. Report Q with df and p, I², tau², weights, prediction interval, exclusions and leave-one-database-out recomputation. A prediction interval needs at least three independent databases.",
                 "逆方差固定效应和 REML 随机效应；可选原始 Hartung-Knapp t 区间，DerSimonian-Laird 仅作为标注的敏感性分析。比值使用对数系数及其标准误。报告 Q 及自由度和 p 值、I²、tau²、权重、预测区间、排除原因和逐库剔除重算结果。预测区间至少需要三个独立数据库。")[language]]
    notes = METHOD_NOTES.get(row["id"])
    if notes is None:
        return []
    return [
        notes.execution[language],
        "", ("### Estimators", "### 估计方法")[language],
        *(f"- {line[language]}" for line in notes.estimators),
        "", ("### Reading the results", "### 解读要点")[language],
        *(f"- {line[language]}" for line in notes.interpretation),
        "", ("### Methodological literature", "### 方法学文献")[language],
        *(f"- {citation}" for citation in notes.literature),
    ]


def _validate_method_notes() -> None:
    workflows = {row.skill_id for row in METHOD_SKILLS if row.layer == "research_workflow"}
    if workflows != set(METHOD_NOTES):
        raise RuntimeError(
            "method Skill notes drift: "
            f"missing={sorted(workflows - set(METHOD_NOTES))!r}, "
            f"unknown={sorted(set(METHOD_NOTES) - workflows)!r}"
        )


_validate_method_notes()


# These are translations of named diagnostics, not new validation rules.
_DIAGNOSTIC_LABELS = {
    "fixed-effect CIs": ("Fixed effects with confidence intervals", "固定效应及置信区间"),
    "variance components and ICC at time zero for LMM": ("Variance components and ICC at time zero", "方差分量及零时点 ICC"),
    "robust SE and QIC availability for GEE": ("Robust GEE standard errors and QIC availability", "GEE 稳健标准误及 QIC 可用性"),
    "informative death dropout": ("Informative dropout from death", "死亡所致信息性脱落"),
    "denominators": ("Counts used as denominators", "分母所用人数"),
    "measurement availability": ("Which measurements are available", "哪些测量可用"),
    "empirical frequency reconciliation": ("Agreement between counts and frequencies", "人数与频率是否一致"),
    "closed exposure/outcome levels and denominators": ("Exposure and outcome groups account for all patients", "暴露与结局分组是否覆盖全部患者"),
    "interval/dependence contract": ("Uncertainty calculation matches repeated observations", "区间计算是否考虑重复观察"),
    "descriptive-only noncausal claim ceiling": ("Interpretation stays descriptive", "解释是否保持描述性"),
    "model-term coding receipt": ("Recorded variable coding", "记录的变量编码"),
    "primary model contract": ("Fitted model matches the reviewed model", "拟合模型是否与审阅后的模型一致"),
    "effect/interval reconciliation": ("Agreement of estimates and uncertainty intervals", "估计值与区间是否一致"),
    "binary outcome proportion and covariance method for RR": ("Outcome frequency and the variance method behind a risk ratio", "结局发生比例及风险比（RR）的方差计算方法"),
    "declared levels": ("Prespecified exposure levels", "预设暴露等级"),
    "primary contrast": ("Prespecified main comparison", "预设主要比较"),
    "model contract": ("Model matches the reviewed specification", "模型是否与审阅后的设定一致"),
    "landmark eligibility and observation opportunity": ("Eligibility and observation at the landmark", "固定时间点的纳入条件和观察机会"),
    "landmark eligibility and exposure opportunity": ("Eligibility and exposure observation at the landmark", "固定时间点的纳入条件及暴露观察机会"),
    "observed knots and reference": ("Spline knots and reference within observed values", "样条结点及参考值是否在观察范围内"),
    "spline-versus-linear likelihood-ratio comparison": ("Spline versus linear model comparison", "样条与线性模型比较"),
    "complete-case model population": ("Patients retained in the complete-case model", "完整案例模型实际保留的患者"),
    "retained early events and unmeasured states": ("Early events and unmeasured states retained", "是否保留早期事件和未测量状态"),
    "patient-cluster count": ("Number of distinct patients", "独立患者数"),
    "convergence and finite clustered covariance": ("Model convergence and finite patient-clustered uncertainty", "模型收敛及患者聚类区间计算是否有效"),
    "event/censoring closure": ("Event and censoring definitions", "事件与删失定义"),
    "time-origin binding": ("Follow-up starts at the reviewed time", "随访是否从审阅后的时间起点开始"),
    "proportional-hazards diagnostic when Cox is declared": ("Proportional-hazards assessment for Cox", "Cox 模型的比例风险评估"),
    "explicit competing-event estimands and supplemental product digests": ("Competing-event definition with its cumulative incidence, Gray test and Fine–Gray results", "竞争事件定义及其累积发生率、Gray 检验和 Fine–Gray 结果"),
    "patient split/leakage": ("Patients do not overlap across data partitions", "不同数据集的患者是否重叠"),
    "development-only preprocessing": ("Preprocessing learned only from development patients", "预处理是否仅从开发患者学习"),
    "discrimination with interval": ("Discrimination and its uncertainty interval", "区分度及其区间"),
    "calibration and Brier score": ("Calibration and Brier score", "校准及 Brier 分数"),
    "prediction-time/observation-window/target-horizon separation": ("Predictors precede the future outcome window", "预测变量是否早于未来结局窗口"),
    "patient-level split and leakage audit": ("Patient separation and information leakage review", "患者划分及信息泄漏检查"),
    "per-landmark discrimination, calibration and horizon observability": ("Performance and observable follow-up at each landmark", "逐时点评估性能及随访是否可观察"),
    "temporal subgroup or drift assessment when data support it": ("Change over calendar time when supported by data", "数据支持时检查性能随日历时间的变化"),
    "target-trial time-zero and treatment-strategy protocol": ("Alignment of eligibility, treatment and follow-up", "纳入条件、治疗与随访是否对齐"),
    "identification/refutation": ("Causal assumptions and checks that could challenge them", "因果假设及可能挑战这些假设的检查"),
    "positivity and balance": ("Treatment overlap and covariate balance", "治疗组重叠及协变量平衡"),
    "representation": ("Clinical features and their representation", "临床特征及其表示方法"),
    "cluster stability": ("Stability of patient groups", "患者分组稳定性"),
    "descriptive outcome use": ("Outcomes only describe groups after clustering", "结局是否仅用于聚类后的分组描述"),
    "fixed-window representation and missingness": ("Feature representation and missingness within the selected window", "所选窗口内的特征表示和缺失"),
    "closed candidate-grid BIC decision": ("BIC selection within the prespecified candidate models", "是否在预设候选模型中按 BIC 选择"),
    "resampling stability or explicit failed-closed non-solution": ("Resampling stability or a recorded inability to find a solution", "重采样稳定性，或明确记录未能得到解"),
}


@dataclass(frozen=True, slots=True)
class PackageStop:
    owner: str
    code: str
    situation: Bilingual
    next_step: Bilingual


PACKAGE_STOPS = (
    PackageStop("typed_input_binding", "digest_mismatch",
                ("The analysis data differ from the reviewed copy.", "分析数据与审阅时的版本不同。"),
                ("Ask Copilot to refresh the data record and prepare the updated plan for review.", "请 Copilot 更新数据记录并准备修订后的计划供审阅。")),
    PackageStop("typed_input_binding", "contract_columns_mismatch",
                ("The supplied table does not contain the agreed columns.", "提供的表格与计划所需列不一致。"),
                ("Review the proposed extraction correction or revised variable list before rerunning.", "重新运行前，审阅建议修正的数据提取方案或变量清单。")),
    PackageStop("question_requirements", "question_requirement_not_covered",
                ("The plan leaves part of your question unanswered.", "计划遗漏了问题中的一部分。"),
                ("Review the revised plan that includes that comparison or explicitly narrows the question.", "审阅补上该比较的计划，或审阅明确缩小问题范围后的计划。")),
    PackageStop("question_requirements", "question_requirement_capability_gap",
                ("The requested analysis is not supported for this plan.", "当前计划无法支持所请求的分析。"),
                ("Review the stated limitation and the proposed sibling workflow or data change before proceeding.", "审阅说明的限制及建议改用的研究方法或数据方案，再决定是否继续。")),
    PackageStop("question_requirements", "question_requirements_unreadable",
                ("The record of the reviewed question cannot be read.", "无法读取已审阅问题的记录。"),
                ("Ask Copilot to prepare a fresh complete plan for review.", "请 Copilot 重新准备完整计划供审阅。")),
)


def validate_package_stops(stops: tuple[PackageStop, ...]) -> None:
    from .execution.runners.typed_input_binding import BINDING_REASON_CODES
    from .planning.question_requirements import QUESTION_REQUIREMENT_STOP_CODES

    registries = {"question_requirements": QUESTION_REQUIREMENT_STOP_CODES,
                  "typed_input_binding": BINDING_REASON_CODES}
    for stop in stops:
        if stop.code not in registries.get(stop.owner, ()):
            raise RuntimeError(f"skill.stop_code_invalid: {stop.owner}/{stop.code}")


def _workflow_guide(row: dict[str, Any]) -> WorkflowGuide:
    return WORKFLOW_GUIDES[row["id"]]


def package_database_notes(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Project db6a for the explicit illustration; never infer actual run readiness.

    Uses the same assess_study/profile owners as webserver.research_capability_scope.
    The profile contains no real-data acceptance receipt, so absence stays unknown.
    """
    if row["id"] not in WORKFLOW_GUIDES:
        return []
    requirements = _workflow_guide(row).requirements
    return [dict(display_name="eICU-CRD" if key == "eicu" else get_database_profile(key).display_name,
                 decision=assess_study(research_capability_profile(key), requirements).model_dump(mode="json"),
                 real_data_acceptance="not_verified") for key in public_database_keys()]


_DATABASE_TERMS = {
    "hospital_death": ("death before hospital discharge", "出院前死亡"),
    "icu_death": ("death before ICU discharge", "出 ICU 前死亡"),
    "death_within_72h_after_icu_discharge": ("death within 72 hours after ICU discharge", "出 ICU 后 72 小时内死亡"),
    "recorded_death_time": ("recorded death time", "记录的死亡时间"),
    "date": ("date of death without a precise time", "仅有死亡日期，无精确时刻"),
    "last_observation_proxy": ("last observation time used as a proxy for death time", "以最后观察时间近似死亡时间"),
    "icu_discharge_proxy": ("ICU discharge time used as a proxy for death time", "以 ICU 出科时间近似死亡时间"),
    "icu_stay": ("ICU stay", "ICU 住院"), "hospitalization": ("hospital admission", "医院住院"),
    "event_time": ("event time", "事件时间"), "censor_time": ("end of follow-up", "随访结束时间"),
    "time_zero": ("ICU admission time", "入 ICU 时间"), "analysis_unit": ("study unit identifier", "分析单位标识"),
    "treatment_timing": ("treatment timing", "治疗时间"), "observation_timing": ("measurement times", "测量时间"),
    "mortality_status": ("death status", "死亡状态"), "hospital_los": ("hospital length of stay", "医院住院时长"),
    "icu_los": ("ICU length of stay", "ICU 住院时长"), "age": ("age", "年龄"), "weight": ("weight", "体重"),
    "sex": ("sex", "性别"), "lact": ("lactate", "乳酸"), "hr": ("heart rate", "心率"),
    "map": ("mean arterial pressure", "平均动脉压"), "norepi_rate": ("noradrenaline dose rate", "去甲肾上腺素用药速率"),
}


def _database_term(value: Any, language: int) -> str:
    if isinstance(value, dict):
        return f"{_database_term(value['concept'], language)} {value['operator']} {value['value']:g}"
    token = str(value).removeprefix("concept:").removeprefix("threshold:")
    return _DATABASE_TERMS.get(token, ("a requested definition needing review", "一项需要审阅的定义"))[language]


def _database_lines(row: dict[str, Any], language: int) -> list[str]:
    if row["id"] not in WORKFLOW_GUIDES:
        return [("The enclosing research workflow owns the per-database assessment. Review its proposed population, variables and outcome; this supporting method does not define a separate source scope.",
                 "数据库评估由所属研究流程负责。请审阅其提出的人群、变量和结局；此辅助方法不另行定义数据适用范围。")[language]]
    lines = [("For the public-demo plan illustrated in examples/public_demo.md; the actual plan is reassessed. Exact means the requested definitions are mapped, not that the study has run.",
              "以下对应 examples/public_demo.zh.md 中的公共演示计划；实际计划会重新评估。“精确对应”仅表示所需定义已有映射，不表示研究已经运行。")[language]]
    for note in package_database_notes(row):
        decision = note["decision"]
        status = {"exact": ("Exact", "精确对应"), "substitution": ("Substitution proposed", "建议替代定义"),
                  "refusal": ("Not available", "不可用")}[decision["status"]][language]
        details = [f"{_database_term(s['original'], language)} → {_database_term(s['substitute'], language)}"
                   for s in decision["substitutions"]]
        details += [("Missing: ", "缺少：")[language] + _database_term(m["capability"], language) for m in decision["missing"]]
        details += [("Still unassessed: ", "尚未评估：")[language] + _database_term(c, language)
                    for c in decision["unassessed_concepts"]]
        details.append(("Real-data acceptance not verified by the source profile.", "来源资料未证明已通过真实数据验收。")[language])
        lines.append(f"- **{note['display_name']} — {status}**. " + "; ".join(details))
    lines.append(("Any substitution changes the definition and requires review of a revised plan. Mapping does not establish observed coverage or adequate sample size.",
                  "替代会改变定义，需要审阅修订后的计划。存在映射不代表实际记录完整，也不代表样本量足够。")[language])
    lines.append(("These source facts do not verify local data access, landmark eligibility or whether a model can be fitted.",
                  "这些来源信息不验证本地数据访问、固定时间点的入组条件或模型能否拟合。")[language])
    return lines


@dataclass(frozen=True, slots=True)
class PackageStep:
    skill_id: str
    action_ids: tuple[str, ...]
    capability_id: str
    title: Bilingual
    checks: tuple[str, ...]
    validator_owner: str | None


def package_steps(row: dict[str, Any]) -> tuple[PackageStep, ...]:
    by_id = {item.skill_id: item for item in METHOD_SKILLS}
    steps = []
    for ident in (*row.get("included_module_ids", ()), row["id"]):
        spec = by_id.get(ident)
        if not isinstance(spec, MethodSkillSpec):
            continue
        capability = _CAPABILITY_BY_ID[spec.capability_id]
        actions = spec.action_ids
        # The registry owns this complete primary operation, including actions
        # without a standalone adapter. Do not manufacture a script or a check.
        steps.append(PackageStep(ident, actions, spec.capability_id,
                                 (spec.title, spec.title_zh), capability.required_diagnostics,
                                 capability.scientific_validator_owner))
    return tuple(steps)


def _step_lines(row: dict[str, Any], language: int) -> list[str]:
    lines = []
    for index, step in enumerate(package_steps(row), 1):
        checks = "; ".join(_DIAGNOSTIC_LABELS[key][language] for key in step.checks)
        label = (("Host checks", "系统检查") if step.validator_owner else
                 ("Required diagnostic evidence for review", "须审阅的诊断记录"))[language]
        lines.append(f"{index}. **{step.title[language]}**. {label}: {checks}.")
        for action in step.action_ids:
            method = next(m for suite in METHOD_SUITE_REGISTRY for m in suite.methods
                          if f"{suite.family}.{m.key}" == action)
            lines.append(f"   - {method.name if language == 0 else _COMPONENT_ZH[action][0]}")
    if not lines:
        lines = [("1. Run the reviewed method within the study; retain its result and reporting evidence.",
                  "1. 在研究中运行审阅后的方法，保留结果与报告所需记录。")[language]]
    return lines


def _reference_lines(row: dict[str, Any], language: int) -> list[str]:
    action_ids = {a for step in package_steps(row) for a in step.action_ids}
    if row.get("method_family"):
        action_ids.add(row["id"])
    lines = []
    for suite in METHOD_SUITE_REGISTRY:
        for method in suite.methods:
            action = f"{suite.family}.{method.key}"
            if action not in action_ids:
                continue
            for item in method.reporting_items:
                # The phenotyping items are internal, not a consensus guideline.
                if item.startswith("internal_phenotype"):
                    continue
                title = method.name if language == 0 else _COMPONENT_ZH[action][0]
                lines.append(f"- {item}: {title}.")
    if row.get("category") in {"Association", "Descriptive", "Survival"}:
        lines.append(("- RECORD: source and variable definitions must accompany the descriptive and association outputs; this package alone does not complete the checklist.",
                      "- RECORD：描述和关联结果须附上来源与变量定义；仅凭此包不能完成整份清单。")[language])
    if row["id"] == "fixed-landmark-association-study":
        lines.append("- STROBE 13 / RECORD 13.1: `cohort_flow.csv`; STROBE 16: `adjusted_association_estimates.csv`.")
    if row["id"] == "target-trial-emulation":
        lines.append(("- Target-trial emulation framework: eligibility, strategies, time zero, follow-up, outcome, causal contrast and analysis plan in the reviewed protocol. Balance and overlap address measured confounding only.",
                      "- 目标试验模拟框架：审阅后的方案说明纳入条件、策略、时间零点、随访、结局、因果比较和分析计划。平衡与重叠只涉及已测量的混杂。")[language])
    if row["id"] == "dynamic-landmark-prediction":
        lines.append(("- TRIPOD+AI: report performance and observable follow-up separately for each prediction time; no complete checklist is claimed.",
                      "- TRIPOD+AI：按预测时点分别报告性能及可观察随访；不声称已完成整份清单。")[language])
    if row.get("category") == "Phenotyping":
        lines.append(("- The registered phenotyping reporting items cover representation, selection and stability. They are internal reporting requirements; no consensus phenotyping guideline is registered here.",
                      "- 已登记的表型报告要求涉及表示方法、选择和稳定性。这些是内部报告要求，此处未登记共识表型报告指南。")[language])
    return list(dict.fromkeys(lines)) or [("- The enclosing study selects the applicable reporting guideline.", "- 由所属研究确定适用的报告指南。")[language]]


_PACKAGE_HEADINGS = (
    ("Purpose", "用途"), ("When to use", "何时使用"), ("When not to use", "何时不适用"),
    ("How it works", "工作原理"), ("Methods", "方法"),
    ("What the plan will propose for your review", "计划将提出哪些内容供您审阅"),
    ("Steps and their checks", "步骤与检查"), ("Outputs", "输出"), ("Per-database notes", "各数据库说明"),
    ("Common stops", "常见停止原因"), ("References", "参考依据"), ("Boundary", "使用边界"),
)


def _output_lines(row: dict[str, Any], language: int) -> list[str]:
    if row["id"] not in WORKFLOW_GUIDES:
        return [("The result of this method, its diagnostic records and the study's selected tables or figures.",
                 "此方法的结果、诊断记录，以及研究选定的表格或图形。")[language]]
    guide = _workflow_guide(row)
    supporting = ("table_one.csv",) if row["id"] in REFERENCE_SCRIPT_PACKAGES else ("table_one.csv", "missingness_summary.csv")
    files = (*supporting, *guide.output_files)
    lines = ["- `" + name + "`" for name in dict.fromkeys(files)]
    if guide.planned_filenames:
        lines.append(("Primary filenames above are proposed for this example plan; they are not fixed exports of a packaged executor. Review the complete output list in the plan.",
                      "以上主要结果文件名是本示例计划的建议，并非内置执行程序固定导出的文件名。请在计划中审阅完整输出清单。")[language])
    lines.append(("These are expected files, not recorded results. Outputs are produced only after a successful reviewed run.",
                  "这些是预期文件，并非已记录的结果；只有审阅后的运行成功才会生成。")[language])
    return lines


def _package_skill_markdown(kind: str, row: dict[str, Any], *, language: int = 0) -> str:
    guide = _workflow_guide(row) if kind == "research_workflow" else None
    validate_package_stops(PACKAGE_STOPS)
    title = row["title_zh"] if language else row["title"]
    description = row["description_zh"] if language else row["description"]
    by_id = {item.skill_id: item for item in METHOD_SKILLS}
    exclusions = []
    for item in row.get("not_for", ()):
        sibling = by_id.get(item["sibling_skill_id"])
        alternative = (sibling.title_zh if language else sibling.title) if sibling else (
            "No supported sibling; review the limitation before proceeding.", "暂无可替代方法，请先审阅这一限制。"
        )[language]
        exclusions.append(f"- {item['situation_zh' if language else 'situation_en']} → {alternative}")
    choices = guide.decisions if guide else (
        "the applicable variables, windows, comparisons and missing-data handling for this method",
        "此方法适用的变量、时间窗、比较与缺失数据处理",
    )
    decisions = (f"The plan proposes {choices[0]}; you review them together before anything runs. Copilot resolves routine configuration and presents a complete plan.",
                 f"计划将提出{choices[1]}；您在运行前一次审阅。Copilot 处理常规配置并呈现完整计划。")[language]
    philosophy = guide.philosophy if guide else (
        "This method answers one part of the reviewed question. The study plan determines the population and scientific choices. Its outputs and checks remain part of that study and do not create a separate primary question.",
        "此方法回答已审阅问题中的一部分。研究计划决定人群与科学选择。其输出和检查属于该研究，不会另行产生主要研究问题。",
    )
    inputs = ("Inputs: an approved cohort with patient/stay identifiers, the specified measurements, units and timestamps, and outcome/follow-up information when used.",
              "输入：获准使用的队列，含患者及住院标识、所需测量、单位和时间戳，以及分析所用的结局与随访信息。")[language]
    if row["id"] == "cross_database.meta_analysis":
        inputs = ("Inputs: sealed per-database aggregate estimates, coefficient SEs, exact estimand identities, capability decisions and each run's passed gates. No patient rows or identifiers are pooling inputs.",
                  "输入：各数据库已封存的汇总效应、系数标准误、精确估计目标标识、能力判定及各自通过的审核。合并不接收患者行或患者标识。")[language]
    boundary = ("EasyICU owns data access, execution, evidence records and the limits on conclusions. Review the full plan before running. Do not change definitions or fill gaps silently; preserve failed checks and revise the plan. Public demos support learning, not clinical conclusions.",
                "EasyICU 负责数据访问、执行、证据记录和结论范围。运行前审阅完整计划；不能悄悄更改定义或填补缺口，须保留未通过的检查并修订计划。公共演示数据用于学习，不能支持临床结论。")[language]
    ceiling = (("Results can enter a report only after the host's required reviews pass.", "结果只有通过系统要求的审查后才能用于报告。")
               if row.get("claim_ceiling") == "reportable" else
               ("Results remain exploratory analysis; this package cannot authorize manuscript findings.", "结果仅供探索分析，此包不能据此授权论文结论。"))[language]
    sections = [
        [description],
        [("Use this for questions such as:", "适合回答此类问题：")[language], guide.question[language]]
        if kind == "research_workflow" else [description,
         ("Use this as a supporting method in a reviewed study.", "作为已审阅研究中的辅助方法使用。")[language]],
        exclusions or [("- A complete study needs a suitable research workflow around this method.", "- 完整研究需要选择包含此方法的研究流程。")[language]],
        [philosophy[language]], _method_lines(row, language), [decisions, inputs], _step_lines(row, language),
        _output_lines(row, language), _database_lines(row, language),
        [f"- **{stop.situation[language]}** {stop.next_step[language]}" for stop in PACKAGE_STOPS],
        _reference_lines(row, language), [boundary, ceiling],
    ]
    if row["id"] in REFERENCE_SCRIPT_PACKAGES:
        sections[6].extend([
            ("### Reference scripts (executable)", "### 参考脚本（可执行）")[language],
            ("The reference implementation loads the cohort, runs the analysis, draws figures and exports a checked report. The existing completion messages include `✓ Analysis completed successfully!` and `=== Export Complete ===`; the latter appears only after export consistency checks pass.",
             "参考实现依次读取队列、分析、绘图并导出经过检查的报告。已有完成消息包括 `✓ Analysis completed successfully!` 和 `=== Export Complete ===`，后者仅在导出一致性检查通过后出现。")[language],
        ])
    if row["id"] == "trajectory-phenotyping":
        sections[6].append(("### Trajectory discovery and early assignment", "### 轨迹发现与早期识别")[language])
        sections[6].append(guide.philosophy[language])
    lines = ["---", f"name: {json.dumps(row['id'])}", f"description: {json.dumps(description, ensure_ascii=False)}",
             f"category: {json.dumps(row['category_zh'] if language else row['category'], ensure_ascii=False)}", "---", "", f"# {title}"]
    for heading, content in zip(_PACKAGE_HEADINGS, sections, strict=True):
        if content:
            lines.extend(["", f"## {heading[language]}", "", *content])
    return "\n".join(lines) + "\n"


def _package_contract_markdown(kind: str, row: dict[str, Any]) -> str:
    """Keep machine coordinates in the Files tab's contract, out of the guide."""
    notes = json.dumps(package_database_notes(row), ensure_ascii=False, sort_keys=True).encode()
    payload = {key: row[key] for key in ("id", "capability_id", "action_ids", "included_module_ids",
                                       "execution_mode", "claim_ceiling", "kernel_modules") if key in row}
    payload.update(package_version=METHOD_SKILL_PACKAGE_VERSION, kind=kind,
                   steps=[dict(skill_id=step.skill_id, action_ids=step.action_ids,
                               checks=step.checks, validator_owner=step.validator_owner) for step in package_steps(row)],
                   common_stops=[dict(owner=stop.owner, code=stop.code) for stop in PACKAGE_STOPS],
                   example_requirements=(_workflow_guide(row).requirements.model_dump(mode="json")
                                         if kind == "research_workflow" else None),
                   database_assessment_sha256=hashlib.sha256(notes).hexdigest())
    return "# Package contract\n\n```json\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n```\n"


def _package_validation_markdown(row: dict[str, Any]) -> str:
    return "# Checks / 检查\n\n" + "\n".join(_step_lines(row, 0) + [""] + _step_lines(row, 1)) + "\n"


def _package_composition_markdown(row: dict[str, Any]) -> str:
    by_id = {item.skill_id: item for item in METHOD_SKILLS}
    return "# Included methods / 包含的方法\n\n" + "\n".join(
        f"- {by_id[ident].title} / {by_id[ident].title_zh}" for ident in row.get("included_module_ids", ())
    ) + "\n"


def _package_example_markdown(row: dict[str, Any], language: int) -> str:
    guide = _workflow_guide(row)
    source = {"miiv_demo": "MIMIC-IV demo", "eicu_demo": "eICU demo"}[guide.demo_source]
    return "\n\n".join([
        (f"# Worked example: {source}", f"# 完整示例：{source}")[language],
        (f"Public source: {source}. This is a proposed teaching plan, not a recorded run; no result values or successful execution are claimed.",
         f"公共来源：{source}。这是拟议的教学计划，并非已记录的运行；不声称有结果数值或已成功执行。")[language],
        guide.question[language],
        ("## Plan proposed for your review", "## 供您审阅的计划")[language], guide.example_plan[language],
        ("## Expected files", "## 预期文件")[language], "\n".join(_output_lines(row, language)),
        ("If the demo lacks the required observations, events or patient separation, the host records the limitation. Review a revised plan; do not fabricate outputs or treat a demo as validation on real data.",
         "若演示数据缺少所需观察、事件或独立患者，系统将记录限制。请审阅修订后的计划，不能补造输出或将演示当作真实数据验证。")[language],
    ]) + "\n"


def _trajectory_assignment_markdown() -> str:
    return """# Trajectory discovery and early subtype assignment

## Stage 1: discover and freeze subtypes

- Fix the trajectory observation window, variables, representation, candidate grid,
  selection rule, and stability rule before comparing solutions.
- Freeze the selected solution and its label mapping. Outcomes may describe frozen
  groups, but must not enter the clustering feature set.

## Stage 2: identify frozen subtypes from early features

- End the clinical feature window before the phenotype-defining trajectory window.
- Keep development and validation patients disjoint.
- Fit preprocessing only on development patients.
- Predict the frozen labels; do not recluster, merge, or rename them during modelling.
- Report confusion, balanced accuracy, macro-F1, log loss, multiclass Brier score,
  per-subtype calibration, and the exact feature roster.

This stage estimates subtype assignment performance. It is distinct from predicting a
clinical outcome from trajectory shape and remains analysis-only without independent
external validation.
"""


def _trajectory_assignment_python() -> str:
    return '''\
"""Reviewed entrypoint for early-feature assignment to frozen trajectory subtypes."""

from easyicu.research_agent.methods.subtype_assignment import (
    fit_and_evaluate_early_subtype_assignment,
)


def run(
    development_features,
    development_labels,
    validation_features,
    validation_labels,
    *,
    development_patient_ids,
    validation_patient_ids,
    feature_window_end,
    phenotype_window_start,
    forbidden_feature_names=(),
):
    return fit_and_evaluate_early_subtype_assignment(
        development_features,
        development_labels,
        validation_features,
        validation_labels,
        development_patient_ids=development_patient_ids,
        validation_patient_ids=validation_patient_ids,
        feature_window_end=feature_window_end,
        phenotype_window_start=phenotype_window_start,
        forbidden_feature_names=forbidden_feature_names,
    )
'''


def _reference_script_sources(package: str) -> dict[str, str]:
    """Read one executable skill package's SKILL.md, references and scripts.

    Files are read from disk rather than imported so that projecting the
    catalogue never imports statsmodels/matplotlib; the package's own tests
    prove the scripts run.
    """

    root = Path(__file__).resolve().parent / "skill_packages" / package
    if not root.is_dir():
        raise MethodSkillPackageNotFound(
            f"reference script package {package!r} is missing from the checkout"
        )
    sources: dict[str, str] = {}
    skill_markdown = root / "SKILL.md"
    if skill_markdown.is_file():
        sources["references/reference_scripts.md"] = skill_markdown.read_text(
            encoding="utf-8"
        )
    for path in sorted((root / "references").glob("*.md")):
        sources[f"references/{path.name}"] = path.read_text(encoding="utf-8")
    spec_module = root / "spec.py"
    if spec_module.is_file():
        sources["scripts/spec.py"] = spec_module.read_text(encoding="utf-8")
    for path in sorted((root / "scripts").glob("*.py")):
        if path.name == "__init__.py":
            continue
        sources[f"scripts/{path.name}"] = path.read_text(encoding="utf-8")
    if not any(name.startswith("scripts/") for name in sources):
        raise MethodSkillPackageNotFound(
            f"reference script package {package!r} ships no scripts"
        )
    return sources


CLINICIAN_SKILL_IDS = frozenset({
    "six-database-cohort-extraction", "clinical-concept-definitions",
    "treatment-patterns", "sample-size-and-power",
})

_INSTRUCTIONS = {
    "six-database-cohort-extraction": """
## When to use
Choose one of the six registered ICU databases, cohort filters, modules and
export format in Data Extraction. `extraction_projection(database)` projects
the existing export roster and each concept's dictionary availability.
The use action opens that owner with modules, database and Parquet prefilled;
it never submits a job. Choose a registered source and review filters there.
The existing entry is POST /api/jobs/extract; easyicu.api.extraction owns export.

## Not for -> sibling
- Understanding a score or changing a definition -> clinical-concept-definitions.
- Comparing treatment use -> treatment-patterns, after governed extraction.
- Evidence synthesis across databases is outside this skill.

## Guardrails
No new extraction engine, patient read, export, or execution is performed by
this projection. A mapping does not prove observed coverage. Unknown database
identities stop. Source permission and cohort validation remain with Data Extraction.
This layer has no scientific claim ceiling because it produces no finding.
""",
    "clinical-concept-definitions": """
## When to use
Inspect SOFA-2, canonical Sepsis-3, KDIGO AKI or any other registered composite
before choosing variables. Call clinical_definitions(concepts, database=...)
for components, callback, owner time defaults, clinical contract and per-database
availability. With no concepts it projects all registered composites.
AKI profile payloads preserve reference/source-native differences and windows.
Owner source bindings disclose database mappings and alternatives; they do not
authorize substitution. Definition time anchors and study observation windows
are separate. Unspecified study windows require explicit selection.

## Not for -> sibling
- Exporting these variables -> six-database-cohort-extraction.
- Testing association -> adjusted-exposure-outcome-study.
- SOFA-2 sepsis sensitivity is not canonical Sepsis-3; inspect its opt-in contract.

## Guardrails
Metadata comes directly from the dictionary, score and clinical-contract owners.
Mapping-only conformance is not clinical validation. A structurally absent organ
domain cannot be imputed into a SOFA-2 total. Missing dependencies and owner
limitations stay visible. No score is computed and no claim ceiling is assigned.
""",
    "treatment-patterns": """
## When to use
Describe recorded drug-class exposure during the ICU stay by reviewed groups,
and time from ICU admission to first recorded use. Bind a governed prepared
cohort with one row per stay, a reviewed TableOneSpec and a source receipt.
Confirm class-agent coverage, the meaning of no record and complete ICU follow-up.
The class membership comes from the treatment capture registry.

## Not for -> sibling
- Treatment effects or time-dependent confounding -> target-trial-emulation.
- Exposure-outcome association -> adjusted-exposure-outcome-study.
- Sample-size planning -> sample-size-and-power.

## Guardrails and output
Use summarize_treatment_patterns; its Table 1 owner supplies counts/proportions
and median/IQR, and its existing Aalen-Johansen owner supplies first-use CIF.
Proportions use all ascertained stays. Timing medians use only exposed stays:
they are not population time-to-event medians. Death and alive ICU discharge
before use compete because neither allows later in-ICU use. Incomplete stays,
conflicting exit flags and use/exit ties stop. A group with no exposures has no
fitted CIF and an explicit reason.
Time-zero exposure is first recorded use, not proof of a new treatment start.
Missing capture never becomes untreated. Duplicate stays, incomplete class
capture and invalid times stop. Table 1 inferential columns are not returned.
All products remain analysis_only; no causal or manuscript finding is authorized.
""",
    "sample-size-and-power": """
## When to use
Use numeric assumptions supplied by the researcher or a governed aggregate with
its receipt. The host functions accept scalars; no file, database or patient
loader is accepted. Inputs, method, software version, integer sample sizes and
analysis_only status travel with every result. Round each group upward.

- two_proportions(p1, p2): independent groups, two-sided Cohen arcsine h normal
  approximation through statsmodels NormalIndPower. This is not a pooled Wald
  risk-difference calculation; rare events may need a different design.
- two_means(difference, standard_deviation): common-SD independent two-sample
  noncentral t calculation through statsmodels TTestIndPower.
- survival_events(hazard_ratio): Schoenfeld required events under proportional
  hazards; total N needs an assumed event fraction reflecting follow-up/attrition.
- prediction_binary(parameters, cox_snell_r2, prevalence): Riley binary logistic
  development minimum N. Count predictor coefficients, excluding the intercept.

## Not for -> sibling
- An observed treatment comparison -> treatment-patterns.
- Building or validating a fitted model -> clinical-risk-prediction.
- Paired, cluster, noninferiority, competing-risk or non-PH designs need a separate
  reviewed planning method. Continuous/survival prediction is outside this port.
- Post-hoc observed power and manuscript findings are not outputs of this skill.

## Calculation contract
Alpha and power are probabilities with power > alpha; effect cannot be null.
For Schoenfeld, D=ceil((z(1-alpha/2)+z(power))² / [q(1-q) log(HR)²]).
For Riley let R be anticipated adjusted Cox-Snell R², p the candidate parameters,
S the target shrinkage, and M=1-exp(2*[pi*log(pi)+(1-pi)*log(1-pi)]).
Criterion 1: n1=p/[(S-1)*log(1-R/S)]. Criterion 2 uses
S2=R/(R+delta*M) in the same equation, limiting Nagelkerke R² optimism to delta.
Criterion 3: n3=z((1+confidence)/2)²*pi*(1-pi)/margin², the Wald precision
criterion for overall risk. Return ceil(max(n1,n2,n3)), each criterion and
expected events. R must be below M, S and S2; never substitute Nagelkerke R².

## References and validation basis
Statsmodels stats.power NormalIndPower and TTestIndPower (installed version is
recorded); Cohen, Statistical Power Analysis for the Behavioral Sciences, 1988.
Schoenfeld DA, Biometrics 1983;39:499-503, doi:10.2307/2531021.
Riley RD et al., Stat Med 2019;38:1276-1296, doi:10.1002/sim.7992.
This reviewed equation-port surface is restricted to binary prediction, with
the published pmsampsize worked-example inputs pinned in tests. It is not a
vendored or complete pmsampsize implementation. Planning output is never a
manuscript finding and does not certify the assumptions or aggregate receipt.
""",
}

_EXAMPLES = {
    "six-database-cohort-extraction": "extraction_projection('eicu_demo') returns the official demo's inherited mappings and an unsubmitted prefill. No demo patient rows are read.",
    "clinical-concept-definitions": "clinical_definitions(['sofa2', 'sep3', 'kdigo_aki'], database='eicu_demo') shows owner metadata for the public demo schema without opening its patient files.",
    "treatment-patterns": """Public synthetic demo (eight invented stays, no patient data):
```python
from easyicu.research_agent.skill_packages import public_demo, summarize_treatment_patterns
frame, arguments = public_demo()
result = summarize_treatment_patterns(frame, **arguments)
```
Both groups have 2/4 exposed stays; group A first-use median is 3 h (IQR 2.5-3.5 h).
The fixture is synthetic and is not an analysis of an official ICU demo release.""",
    "sample-size-and-power": """Cohen d=.5, alpha=.05, power=.8 gives 64 per arm.
Probabilities .50 vs .65 give 170 per arm under Cohen's arcsine approximation.
Schoenfeld HR=.7 with equal allocation gives 247 required events.
Riley's binary worked example (24 parameters, Cox-Snell R²=.051, prevalence=.174)
gives n=4115 (criteria: 4115, 763, 221), expected events=716.01.
```python
from easyicu.research_agent.skill_packages import prediction_binary
result = prediction_binary(parameters=24, cox_snell_r2=.051, prevalence=.174)
```""",
}

_STOPS = {
    "six-database-cohort-extraction": ["definition.database_unknown"],
    "clinical-concept-definitions": ["definition.database_unknown", "definition.concepts_required", "definition.concept_unknown"],
    "treatment-patterns": ["treatment.class_unknown", "treatment.class_capture_incomplete",
        "treatment.capture_review_required", "treatment.summary_contract_invalid", "treatment.columns_required",
        "treatment.one_row_per_stay_required", "treatment.ascertainment_required", "treatment.time_invalid",
        "treatment.event_order_unresolved", "treatment.no_exposures"],
    "sample-size-and-power": ["planning.invalid_number", "planning.invalid_probability",
        "planning.positive_required", "planning.power_not_above_alpha", "planning.null_effect",
        "planning.assumption_source_required", "planning.aggregate_receipt_required",
        "planning.assumption_source_mismatch", "planning.numerical_failure",
        "planning.invalid_parameter_count", "planning.incompatible_r2"],
}


def clinician_package_sources(row: dict) -> dict[str, str]:
    name = row["id"]
    sources = {
        "SKILL.md": (f"---\nname: {name}\ndescription: {json.dumps(row['description'])}\n"
                     f"category: {json.dumps(row['category'])}\n---\n\n# {row['title']}\n"
                     + _INSTRUCTIONS[name].strip() + "\n"),
        "references/example.md": _EXAMPLES[name] + "\n",
        "references/stops.json": json.dumps({code: code.replace(".", ": ").replace("_", " ")
                                            for code in _STOPS[name]}, indent=2) + "\n",
    }
    if name == "treatment-patterns":
        sources["scripts/run.py"] = (
            "from easyicu.research_agent.skill_packages import "
            "summarize_treatment_patterns as run, public_demo\n"
        )
    elif name == "sample-size-and-power":
        sources["scripts/run.py"] = (
            "from easyicu.research_agent.skill_packages import "
            "two_proportions, two_means, survival_events, prediction_binary\n"
        )
    return sources


def method_skill_workflow_metadata(skill_id: str, *, enabled: bool) -> dict[str, Any]:
    """Read-only structured guide coordinates for host workflow projections.

    This is separate from package generation: consumers need not parse prose
    or know how package files and implementations are assembled.
    """
    kind, row, _ = _package_entry(skill_id, enabled=enabled)
    if kind != "research_workflow":
        raise MethodSkillPackageNotFound(skill_id)
    guide = _workflow_guide(row)
    return {
        "requirements": guide.requirements.model_dump(mode="json"),
        "input_labels": {value: {"label": _database_term(value, 0), "label_zh": _database_term(value, 1)}
                         for value in (*guide.requirements.concepts, guide.requirements.outcome) if value},
        "output_files": list(guide.output_files),
        "planned_filenames": guide.planned_filenames,
        "steps": [dict(id=step.skill_id, action_ids=list(step.action_ids),
                       capability_id=step.capability_id, label=step.title[0], label_zh=step.title[1])
                  for step in package_steps(row)],
    }


def method_skill_package(skill_id: str, *, enabled: bool) -> dict[str, Any]:
    """Return one digest-bound, read-only package grouped by purpose."""

    kind, row, catalog_sha256 = _package_entry(skill_id, enabled=enabled)
    sources = clinician_package_sources(row) if row["id"] in CLINICIAN_SKILL_IDS else {
        "SKILL.md": _package_skill_markdown(kind, row),
        "SKILL.zh.md": _package_skill_markdown(kind, row, language=1),
        "references/validation_framework.md": _package_validation_markdown(row),
    }
    contract_name = (
        "workflow_contract.md"
        if kind == "research_workflow"
        else "module_contract.md"
        if kind == "analysis_module"
        else "method_contract.md"
    )
    if row["id"] not in CLINICIAN_SKILL_IDS:
        sources[f"references/{contract_name}"] = _package_contract_markdown(kind, row)
    if kind == "research_workflow":
        sources["references/composition.md"] = _package_composition_markdown(row)
        sources["examples/public_demo.md"] = _package_example_markdown(row, 0)
        sources["examples/public_demo.zh.md"] = _package_example_markdown(row, 1)
    if row["id"] == "trajectory-phenotyping":
        sources["references/trajectory_assignment.md"] = _trajectory_assignment_markdown()
        sources["scripts/early_subtype_assignment.py"] = _trajectory_assignment_python()
    execution_boundary = DOCUMENTATION_ONLY_BOUNDARY
    if kind == "data_definition":
        execution_boundary = "owner_projection_no_execution"
    if row["id"] in {"treatment-patterns", "sample-size-and-power"}:
        execution_boundary = REFERENCE_SCRIPTS_BOUNDARY
    if row["id"] in REFERENCE_SCRIPT_PACKAGES:
        sources.update(_reference_script_sources(REFERENCE_SCRIPT_PACKAGES[row["id"]]))
        execution_boundary = REFERENCE_SCRIPTS_BOUNDARY
    files = []
    for path, content in sources.items():
        encoded = content.encode("utf-8")
        files.append(
            {
                "path": path,
                "language": "markdown" if path.endswith(".md") else "json" if path.endswith(".json") else "python",
                "size_bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "content": content,
            }
        )
    if kind != "method_component":
        files.extend(method_skill_implementation_files(row["id"], METHOD_SKILLS))
    package_canonical = json.dumps(
        [{"path": row["path"], "sha256": row["sha256"]} for row in files],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": METHOD_SKILL_PACKAGE_VERSION,
        "skill_id": row["id"],
        "kind": kind,
        "title": row["title"],
        "title_zh": row["title_zh"],
        "not_for": row.get("not_for", []),
        "database_notes": package_database_notes(row) if row["id"] not in CLINICIAN_SKILL_IDS else [],
        "overview_paths": {"en": "SKILL.md", "zh": "SKILL.zh.md" if "SKILL.zh.md" in sources else "SKILL.md"},
        "catalog_sha256": catalog_sha256,
        "package_sha256": hashlib.sha256(package_canonical).hexdigest(),
        "read_only": True,
        "execution_boundary": execution_boundary,
        "files": files,
    }


__all__ = [
    "DOCUMENTATION_ONLY_BOUNDARY",
    "METHOD_SKILL_PACKAGE_VERSION",
    "METHOD_SKILL_REGISTRY_VERSION",
    "METHOD_SKILLS",
    "REFERENCE_SCRIPTS_BOUNDARY",
    "REFERENCE_SCRIPT_PACKAGES",
    "MethodSkillPackageNotFound",
    "MethodSkillSpec",
    "method_skill_catalog",
    "method_skill_package",
    "method_skill_workflow_metadata",
]
