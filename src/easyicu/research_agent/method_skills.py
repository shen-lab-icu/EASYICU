"""Public, method-first Skill Hub projection of real EasyICU capabilities.

The research-agent registries own execution and scientific authority.  This
module owns only the user-facing catalogue: a small set of discoverable method
workflows whose capability and action identifiers are checked at import time.
It also projects a read-only package for the desktop Skill Hub, separating the
main instruction, reference contracts, validation rules, and any real
capability-specific helpers. Shared host dispatch stays in the host runtime
instead of being copied into every package.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Tuple

from .planning.capability_registry import CAPABILITY_REGISTRY
from .planning.analysis_method_suite import METHOD_SUITE_REGISTRY
from .planning.method_adapter_catalog import HIGH_FREQUENCY_METHOD_ADAPTERS

METHOD_SKILL_REGISTRY_VERSION = "easyicu.method-skills/3"
METHOD_SKILL_PACKAGE_VERSION = "easyicu.method-skill-package/3"

#: Workflows whose package ships an executable reference implementation from
#: ``research_agent/skill_packages/<package>``: real ``scripts/`` that run the
#: standard workflow through the host kernels, plus the package's own SKILL.md
#: and method notes. Every other workflow is documentation only.
REFERENCE_SCRIPT_PACKAGES: dict[str, str] = {
    "fixed-landmark-association-study": "landmark_categorical_association",
}
DOCUMENTATION_ONLY_BOUNDARY = "documentation_only_host_execution_required"
REFERENCE_SCRIPTS_BOUNDARY = "host_reference_scripts_executable"

ClaimCeiling = Literal["reportable", "analysis_only"]
ExecutionMode = Literal["deterministic_host", "agent_coded_with_host_gates"]
SkillLayer = Literal["research_workflow", "analysis_module"]


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
            "execution_mode": self.execution_mode,
            "claim_ceiling": self.claim_ceiling,
            "scope": capability.primary_estimand,
            "inputs": list(capability.data_contract),
            "outputs": [capability.result_contract],
            "diagnostics": list(capability.required_diagnostics),
            "prompt": self.prompt,
            "prompt_zh": self.prompt_zh,
        }


METHOD_SKILLS: Tuple[MethodSkillSpec, ...] = (
    MethodSkillSpec(
        "cohort-characterization-table-one",
        "Cohort characterization and Table 1",
        "队列描述与 Table 1",
        "Descriptive epidemiology",
        "描述性流行病学",
        "Build a grouped baseline table from one typed, source-bound cohort.",
        "基于一个类型明确、来源绑定的队列生成分组基线特征表。",
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
        "Descriptive epidemiology",
        "描述性流行病学",
        "Audit variable availability, denominators, and declared missingness products.",
        "审计变量可用性、分母及预先声明的缺失数据产物。",
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
        "研究因素与结局分布",
        "Descriptive epidemiology",
        "描述性流行病学",
        "Estimate typed absolute risks and denominators without causal language.",
        "在不使用因果措辞的前提下，计算类型明确的绝对风险与分母。",
        "descriptive_exposure_outcome_distribution_v1",
        (),
        "deterministic_host",
        "reportable",
        "Use the Exposure and outcome distribution method. The research plan proposes the exposure levels, outcome definition, denominators, and interval method; I review them together in the plan, and nothing runs before that review. Interpretation stays descriptive. My research question: ",
        "请使用“研究因素与结局分布”方法。研究因素水平、结局定义、分母与区间方法由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。解释保持描述性，不作因果推断。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "adjusted-association-model",
        "Adjusted association model",
        "调整后关联模型",
        "Association analysis",
        "关联分析",
        "Fit one exact adjusted model with typed terms, references, and transforms.",
        "使用类型、参考水平和变换均已声明的变量拟合一个精确调整模型。",
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
        "Association analysis",
        "关联分析",
        "Evaluate a prespecified ordered exposure with at least three closed levels.",
        "评估至少包含三个封闭水平的预设有序研究因素。",
        "association_ordinal_trend_v1",
        ("association.ordinal_trend",),
        "agent_coded_with_host_gates",
        "analysis_only",
        "Use the Ordinal trend and dose-response method. The research plan proposes the ordered levels and reference, primary contrast, adjustment set, and trend test; I review them together in the plan, and nothing runs before that review. A binary or continuous exposure is not forced into ordered groups. My research question: ",
        "请使用“有序趋势与剂量反应”方法。有序水平与参照、主要对比、调整集与趋势检验由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。不会把二分类或连续研究因素改成有序分组。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "landmark-categorical-association",
        "Landmark categorical association",
        "Landmark 分类关联",
        "Association analysis",
        "关联分析",
        "Estimate a categorical association in a signed fixed-landmark population.",
        "在签名固定 landmark 人群中估计分类研究因素的关联。",
        "association_landmark_categorical_v1",
        ("association.adjusted_association",),
        "deterministic_host",
        "reportable",
        "Use the Landmark categorical association method. The research plan proposes time zero, the landmark, the eligible population, exposure levels and reference, and the adjustment set; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“Landmark 分类关联”方法。时间零点、landmark、合格人群、研究因素水平与参照、调整集由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "landmark-spline-dose-response",
        "Landmark spline dose-response",
        "Landmark 样条剂量反应",
        "Association analysis",
        "关联分析",
        "Model a signed continuous exposure with prespecified knots and reference.",
        "使用预设结点与参考值建模签名连续研究因素。",
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
        "时间变化研究因素 Cox 模型",
        "Survival analysis",
        "生存分析",
        "Run a source-bound counting-process Cox model for a time-updated exposure.",
        "针对时间更新的研究因素运行来源绑定的计数过程 Cox 模型。",
        "association_time_varying_exposure_v1",
        (),
        "deterministic_host",
        "analysis_only",
        "Use the Time-varying exposure Cox model method. The research plan proposes time zero, interval construction, exposure-update rules, event and censoring rules, and covariates; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“时间变化研究因素 Cox 模型”方法。时间零点、区间构造、暴露更新规则、事件与删失规则和协变量由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        layer="analysis_module",
    ),
    MethodSkillSpec(
        "adjusted-exposure-outcome-study",
        "Adjusted exposure-outcome study",
        "观察性研究因素—结局关联",
        "Association analysis",
        "关联分析",
        "Run a complete observational association study from cohort audit through an adjusted primary model.",
        "从队列审计、基线描述和粗分布开始，完成一个调整后的观察性关联研究。",
        "association_adjusted_v1",
        ("association.adjusted_association",),
        "deterministic_host",
        "reportable",
        "Use the Adjusted exposure-outcome study method. The research plan proposes the cohort, time zero, covariates, missing-data strategy, diagnostics, and sensitivity analyses; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“观察性研究因素—结局关联”方法。队列、时间零点、协变量、缺失数据策略、诊断与敏感性分析由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
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
        "Ordinal exposure dose-response study",
        "有序研究因素剂量反应研究",
        "Association analysis",
        "关联分析",
        "Evaluate an ordered exposure across at least three prespecified levels within a complete cohort study.",
        "在完整队列研究中评估至少三个预设有序水平的趋势与剂量反应。",
        "association_ordinal_trend_v1",
        ("association.ordinal_trend",),
        "agent_coded_with_host_gates",
        "analysis_only",
        "Use the Ordinal exposure dose-response study method. The research plan proposes the ordered levels and reference, population, covariates, trend contrast, and nonlinearity checks; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“有序研究因素剂量反应研究”方法。有序水平与参照、人群、协变量、趋势对比与非线性检查由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
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
        "Association analysis",
        "关联分析",
        "Run a complete fixed-landmark study for a categorical exposure without immortal-time leakage.",
        "在避免不死时间泄漏的前提下，完成分类研究因素的固定 landmark 关联研究。",
        "association_landmark_categorical_v1",
        ("association.adjusted_association",),
        "deterministic_host",
        "reportable",
        "Use the Fixed-landmark association study method. The research plan proposes time zero, the landmark, the population still eligible at it, outcome horizon, adjustment set, and censoring; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“固定 Landmark 关联研究”方法。时间零点、landmark、届时仍合格的人群、结局时间范围、调整集与删失由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
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
        "时间变化研究因素生存研究",
        "Survival analysis",
        "生存分析",
        "Run a complete counting-process survival study for a time-updated ICU exposure.",
        "针对时间更新的 ICU 研究因素完成一个计数过程生存研究。",
        "association_time_varying_exposure_v1",
        (),
        "deterministic_host",
        "analysis_only",
        "Use the Time-varying exposure survival study method. The research plan proposes time zero, interval construction, exposure-update rules, event and censoring definitions, patient clustering, and sensitivity analyses; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“时间变化研究因素生存研究”方法。时间零点、区间构造、暴露更新规则、事件与删失定义、患者聚类和敏感性分析由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
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
        "Survival and time-to-event analysis",
        "生存与时间结局分析",
        "Survival analysis",
        "生存分析",
        "Run a contract-bound Cox analysis with KM, log-rank, and PH diagnostics.",
        "运行契约绑定的 Cox 分析，并提供 KM、log-rank 与 PH 诊断。",
        "survival_time_to_event_v1",
        ("time_to_event.cox_hr", "time_to_event.km_logrank", "time_to_event.ph_check"),
        "deterministic_host",
        "reportable",
        "Use the Survival and time-to-event analysis method. The research plan proposes the time origin, event and censoring rules, reference level, covariates, follow-up horizon, and proportional-hazards check; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“生存与时间结局分析”方法。时间起点、事件与删失规则、参照水平、协变量、观察期限与比例风险检验由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
        ),
    ),
    MethodSkillSpec(
        "restricted-mean-survival-time",
        "Restricted mean survival time",
        "限制性平均生存时间（RMST）",
        "Survival analysis",
        "生存分析",
        "Compare two groups over one reviewed RMST horizon.",
        "在一个经审阅的时间范围内比较两组 RMST。",
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
        "Clinical risk prediction",
        "临床风险预测",
        "Prediction modelling",
        "预测建模",
        "Develop and internally validate a leakage-safe binary risk model.",
        "开发并内部验证一个避免数据泄漏的二分类风险模型。",
        "prediction_risk_model_v1",
        ("prediction.discrimination_calibration", "prediction.calibration_metrics", "prediction.decision_curve", "prediction.internal_validation"),
        "deterministic_host",
        "reportable",
        "Use the Clinical risk prediction method. The research plan proposes the prediction time, horizon, predictors, patient-level split, validation, and calibration; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“临床风险预测”方法。预测时点、时间范围、预测变量、患者级数据划分、验证与校准由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
        ),
    ),
    MethodSkillSpec(
        "dynamic-landmark-prediction",
        "Dynamic landmark prediction",
        "动态 Landmark 预测",
        "Prediction modelling",
        "预测建模",
        "Build leakage-safe early-warning predictions at prespecified landmarks.",
        "在预设 landmark 构建避免数据泄漏的早期预警预测。",
        "dynamic_prediction_landmark_v1",
        (),
        "agent_coded_with_host_gates",
        "analysis_only",
        "Use the Dynamic landmark prediction method. The research plan proposes the landmarks, look-back window, horizon, predictors, and patient-level split; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“动态 Landmark 预测”方法。landmark、回看窗口、时间范围、预测变量与患者级数据划分由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
        ),
    ),
    MethodSkillSpec(
        "target-trial-emulation",
        "Target-trial emulation",
        "目标试验模拟",
        "Causal inference",
        "因果推断",
        "Specify a target trial and execute an assumption-bound causal contrast.",
        "明确目标试验方案，并在假设约束下执行因果对比。",
        "causal_target_trial_v1",
        (),
        "agent_coded_with_host_gates",
        "analysis_only",
        "Use the Target-trial emulation method. The research plan proposes eligibility, treatment strategies, time zero, follow-up, causal contrast, and confounding control; I review them together in the plan, and nothing runs before that review. Results stay analysis-only unless the identification and readiness gates pass. My research question: ",
        "请使用“目标试验模拟”方法。纳入标准、治疗策略、时间零点、随访、因果对比与混杂控制由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。除非独立的识别与就绪门禁通过，结果保持仅分析级。我的研究问题是：",
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
        "表型识别",
        "Select and validate a cluster solution without using outcomes as features.",
        "在不把结局作为特征的前提下选择并验证聚类方案。",
        "phenotyping_cluster_v1",
        ("phenotyping.cluster_solution", "phenotyping.k_selection", "phenotyping.cluster_stability", "phenotyping.cluster_sizes", "phenotyping.outcome_by_cluster"),
        "deterministic_host",
        "reportable",
        "Use the Cross-sectional phenotyping method. The research plan proposes the feature window, outcome-free feature list, candidate k grid, and selection and stability rules; I review them together in the plan, and nothing runs before that review. My research question: ",
        "请使用“横断面表型聚类”方法。特征时间窗、不含结局的特征清单、候选 k 网格、选择与稳定性规则由研究计划提出，我在计划里一次审阅，审阅通过前不运行分析。我的研究问题是：",
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
        "表型识别",
        "Discover fixed-window trajectory phenotypes, freeze their labels, and test whether early clinical features can assign new patients to those subtypes.",
        "在固定时间窗发现并冻结轨迹表型，再检验早期临床特征能否识别新患者所属亚型。",
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
        included_module_ids=(
            "cohort-characterization-table-one",
            "missingness-measurement-audit",
        ),
    ),
)


_CAPABILITY_BY_ID = {row.capability_id: row for row in CAPABILITY_REGISTRY}
_ACTION_IDS = {row.action_id for row in HIGH_FREQUENCY_METHOD_ADAPTERS}

_FAMILY_ZH = {
    "time_to_event": "生存分析",
    "causal_emulation": "因果推断",
    "association": "关联分析",
    "prediction": "预测建模",
    "phenotyping": "表型识别",
    "descriptive": "描述性流行病学",
}

_COMPONENT_ZH = {
    "time_to_event.cox_hr": ("Cox 比例风险模型", "估计研究因素与事件风险的调整后风险比。"),
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
    "causal_emulation.negative_control": ("阴性对照分析", "使用预设阴性对照结局或研究因素检查残余偏倚。"),
    "causal_emulation.doubly_robust": ("双重稳健 AIPTW 敏感性", "在点干预、二分类结局条件下估计双重稳健平均处理效应。"),
    "causal_emulation.mediation": ("乘积法中介敏感性分析", "在明确识别假设下分解直接、间接和总效应。"),
    "association.ordinal_trend": ("有序趋势分析", "评估封闭有序水平之间的剂量反应与单调趋势。"),
    "association.adjusted_association": ("调整后关联模型", "拟合参考水平、变量类型和调整集均已声明的关联模型。"),
    "association.multiple_adjustment": ("多层调整模型", "比较粗模型、最小调整和完整调整方案。"),
    "association.effect_modification": ("效应修饰与交互作用", "检查预设修饰因素对应的交互项与亚组结果。"),
    "association.missingness_audit": ("缺失数据与完整案例敏感性", "比较缺失模式、完整案例与预先声明的处理策略。"),
    "association.multiple_testing": ("多重检验校正", "使用 FDR 或 Bonferroni 等预设规则控制多重比较。"),
    "association.evalue": ("E-value 关联敏感性", "评估未测混杂解释调整后关联所需的最低强度。"),
    "association.robustness_panel": ("稳健性规格面板", "汇总不同队列或模型规格下主要估计的稳定性。"),
    "association.rcs_spline": ("限制性立方样条剂量反应", "检查连续研究因素的非线性关联并生成置信区间曲线。"),
    "prediction.discrimination_calibration": ("区分度与校准", "联合评估 AUROC、校准曲线和预测概率的可用性。"),
    "prediction.calibration_metrics": ("校准斜率、截距与 Brier 分数", "定量评估总体校准、校准斜率和概率误差。"),
    "prediction.delong_ci": ("DeLong AUROC 区间与比较", "计算 AUROC 置信区间并比较同一病例上的相关模型。"),
    "prediction.decision_curve": ("决策曲线与净获益", "在临床阈值范围内比较模型、全治和全不治策略。"),
    "prediction.threshold_metrics": ("临床阈值性能", "报告指定阈值下的敏感度、特异度、PPV 和 NPV。"),
    "prediction.feature_attribution": ("特征归因", "使用 SHAP、置换重要性或模型系数解释预测贡献。"),
    "prediction.subgroup_fairness": ("亚组与公平性性能", "比较人口学亚组中的区分度与校准表现。"),
    "prediction.internal_validation": ("内部验证", "用 bootstrap 或交叉验证评估并校正开发集乐观偏倚。"),
    "prediction.conformal_intervals": ("保形预测集", "生成具有边际或分层覆盖保证的预测集合。"),
    "prediction.dynamic_prediction": ("动态 Landmark 预测", "用每个预测时点之前的测量更新未来风险。"),
    "prediction.reclassification": ("NRI 与 IDI 重分类分析", "比较基线模型与更新模型在相同病例上的风险重分类。"),
    "phenotyping.cluster_solution": ("横断面亚表型聚类", "基于早期特征形成并检查患者亚表型。"),
    "phenotyping.k_selection": ("聚类数量选择", "使用 silhouette、gap 或 BIC 等预设标准选择聚类数。"),
    "phenotyping.cluster_stability": ("聚类稳定性与复现性", "通过重采样、共识或调整 Rand 指数检查稳定性。"),
    "phenotyping.trajectory_cluster_stability": ("轨迹聚类稳定性复拟合", "按登记方案重采样并复核已选轨迹模型的稳定性。"),
    "phenotyping.cluster_sizes": ("聚类规模与退化检查", "报告各聚类规模并标记近空或退化聚类。"),
    "phenotyping.outcome_by_cluster": ("聚类间结局描述", "描述冻结聚类之间的临床结局差异，不作因果解释。"),
    "phenotyping.trajectory_feature_clustering": ("轨迹特征聚类", "根据预设时间窗和表示方法构建纵向轨迹表型。"),
    "phenotyping.early_subtype_assignment": ("早期特征识别冻结轨迹亚型", "用轨迹观察窗之前的临床特征识别已冻结的亚型标签，并在患者互斥验证集上评估。"),
    "descriptive.descriptive_summary": ("队列与测量过程描述", "描述研究队列、变量分布和测量过程。"),
    "descriptive.table_one": ("基线特征表（Table 1）", "按组汇总参与者基线特征并保留真实分母。"),
    "descriptive.missingness_audit": ("缺失与完整性审计", "逐变量报告缺失、覆盖和测量可用性。"),
}


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


def _package_skill_markdown(kind: str, row: dict[str, Any]) -> str:
    layer_label = {
        "research_workflow": "complete research workflow",
        "analysis_module": "reusable analysis module",
        "method_component": "method component",
    }[kind]
    claim_text = (
        "The enclosing reviewed workflow may use these results in a report, but "
        "only after every registered diagnostic and evidence gate passes."
        if row["claim_ceiling"] == "reportable"
        else "Results remain analysis-only and cannot be promoted to a paper-level "
        "claim without a separate authority decision."
    )
    section_name = (
        "workflow"
        if kind == "research_workflow"
        else "module"
        if kind == "analysis_module"
        else "method"
    )
    scope_text = str(row.get("scope", row["description"])).strip().rstrip(".")
    lines = [
        "---",
        f"name: {json.dumps(row['id'], ensure_ascii=False)}",
        f"description: {json.dumps(row['description'], ensure_ascii=False)}",
        f"category: {json.dumps(row['category'], ensure_ascii=False)}",
        "---",
        "",
        f"# {row['title']}",
        "",
        str(row["description"]),
        "",
        f"This package is a **{layer_label}** backed by EasyICU's registered host "
        "runtime. It keeps the scientific instructions readable while the host "
        "retains control of data access, review, execution, and evidence receipts.",
        "",
        f"## When to use this {section_name}",
        "",
        f"Use it when the reviewed question requires **{scope_text}**.",
    ]
    if kind in {"research_workflow", "analysis_module"}:
        workflow_name = "workflow" if kind == "research_workflow" else "module"
        lines.extend(
            [
                "",
                f"This {workflow_name} is appropriate when:",
                "",
                f"- {row['description']}",
                "- The source cohort and every requested variable can be bound to a "
                "versioned data source.",
                "- The plan can state the inputs, diagnostics, and outputs below "
                "before execution starts.",
            ]
        )
        if kind == "analysis_module":
            lines.extend(
                [
                    "- The analysis is a supporting step inside a larger study; this "
                    "module does not define a complete project or create a new primary "
                    "estimand by itself.",
                ]
            )
        lines.extend(
            [
                "",
                "## Required data and decisions",
                "",
                *[f"- {value}" for value in row["inputs"]],
                "- Source identity, version, and variable provenance for every bound input.",
                "- Population, exclusions, time windows, units, and denominator rules "
                "whenever they apply to the requested analysis.",
                "",
                "Do not infer any missing scientific choice from column names alone. "
                "Pause at plan review until each required item is explicit.",
                "",
                f"## {'Workflow' if kind == 'research_workflow' else 'Module workflow'}",
                "",
            ]
        )
        detailed_steps = row.get("workflow_steps", [])
        if kind == "research_workflow" and detailed_steps:
            lines.extend(
                f"{index}. {value}"
                for index, value in enumerate(detailed_steps, start=1)
            )
        else:
            lines.extend(
                [
                    "1. **Frame the question.** Translate the request into the registered "
                    f"scope: {row['scope']}.",
                    "2. **Bind the data.** Resolve the cohort, variables, time windows, "
                    "units, reference levels, and provenance without reading beyond the "
                    "approved source boundary.",
                    "3. **Freeze the plan.** Record the registered actions, diagnostics, "
                    "outputs, missing-data rules, and stopping conditions for scientific review.",
                    "4. **Execute through the host.** Run only the registered actions below; "
                    "do not substitute an unreviewed estimator or silently broaden the question.",
                    "5. **Validate before interpretation.** Check every required diagnostic "
                    "and preserve failures in the evidence record.",
                    "6. **Publish bounded outputs.** Return only the registered artifacts and "
                    "interpret them within the declared claim ceiling.",
                ]
            )
        lines.extend(
            [
                "",
                "## Methods and implementation",
                "",
                f"- Capability: `{row['capability_id']}`",
                f"- Skill layer: `{kind}`",
                f"- Execution mode: `{row['execution_mode']}`",
                f"- Package version: `{METHOD_SKILL_PACKAGE_VERSION}`",
                "",
                "### Registered actions",
                "",
                *(
                    [f"- `{value}`" for value in row["action_ids"]]
                    or ["- Bound directly to the registered capability owner."]
                ),
            ]
        )
        if kind == "research_workflow":
            lines.extend(
                [
                    "",
                    "### Included reusable modules",
                    "",
                    *(
                        [f"- `{value}`" for value in row.get("included_module_ids", [])]
                        or ["- No reusable modules are included by default."]
                    ),
                    "",
                    "Included modules support the main workflow. They do not become "
                    "additional primary questions unless the reviewed plan says so.",
                ]
            )
        if row["id"] in REFERENCE_SCRIPT_PACKAGES:
            package = REFERENCE_SCRIPT_PACKAGES[row["id"]]
            lines.extend(
                [
                    "",
                    "### Reference scripts (executable)",
                    "",
                    "This workflow ships a tested reference implementation under "
                    f"`scripts/` (Python package `easyicu.research_agent.skill_packages.{package}`). "
                    "Run it as written instead of writing inline analysis code:",
                    "",
                    "```python",
                    f"from easyicu.research_agent.skill_packages.{package} import (",
                    "    LandmarkCategoricalSpec, load_cohort, run_analysis, generate_all_plots, export_all,",
                    ")",
                    "cohort = load_cohort(cohort_path, spec)        # ✓ Cohort loaded ...",
                    "result = run_analysis(cohort, work_dir=out)    # ✓ Analysis completed successfully!",
                    "figures = generate_all_plots(result, out)      # ✓ All plots generated successfully!",
                    "export_all(result, out, figures=figures)       # === Export Complete ===",
                    "```",
                    "",
                    "Each step prints a verification token; `export_all` prints its token "
                    "only after the export consistency gate passes. Every number in a "
                    "downstream report is copied from `key_metrics.csv`. The package's own "
                    "`references/reference_scripts.md` (its SKILL.md), `references/methods.md`, "
                    "`references/caveat_flags.md`, `references/reporting_checklist.md` and "
                    "`references/comparator_review.md` are included in this package.",
                    "",
                    "Running the scripts does not replace the host's plan review, evidence "
                    "binding or human approval; the output stays analysis-only until the "
                    "host gates say otherwise.",
                ]
            )
        if row["id"] == "trajectory-phenotyping":
            lines.extend(
                [
                    "",
                    "### Trajectory discovery and early assignment",
                    "",
                    "1. Discover candidate trajectory phenotypes only inside the frozen "
                    "phenotype window and select from the closed candidate grid.",
                    "2. Check resampling stability and freeze the subtype definition, "
                    "labels, and label-alignment rule before prediction begins.",
                    "3. Train the assignment model only from features available before "
                    "the phenotype window, using patient-disjoint development and validation sets.",
                    "4. Report phenotype stability and early-assignment performance as "
                    "separate results. Prediction accuracy does not validate the biological "
                    "meaning of the phenotypes.",
                ]
            )
        lines.extend(
            [
                "",
                "## Validation and quality checks",
                "",
                *[f"- {value}" for value in row["diagnostics"]],
                "- Reconcile cohort counts, exclusions, missingness, units, and time "
                "ordering against the frozen plan.",
                "- Stop rather than improvise if a registered action, required input, "
                "or required diagnostic is unavailable.",
            ]
        )
    else:
        modules = row["kernel_modules"] or ["registered host method component"]
        lines.extend(
            [
                "",
                "Use this as one method inside an already framed and reviewed study. "
                "It does not define the cohort, estimand, or complete article workflow on its own.",
                "",
                "## Required inputs and decisions",
                "",
                "- A defined analysis population and versioned source data.",
                "- The variables, time windows, coding, units, comparison or reference, "
                "and missing-data policy required by the method.",
                "- A prespecified output contract and the assumptions that determine "
                "whether the method is applicable.",
                "",
                "## Method and implementation",
                "",
                f"- Method coordinate: `{row['method_family']}.{row['method_key']}`",
                f"- Method tier: `{row['tier']}`",
                f"- Execution mode: `{row['execution_mode']}`",
                f"- Implementation class: `{row['implementation']}`",
                "- Reviewed implementation bindings: "
                + ", ".join(f"`{value}`" for value in modules),
                f"- Package version: `{METHOD_SKILL_PACKAGE_VERSION}`",
                "",
                "The enclosing plan must confirm applicability and assumptions before "
                "this component is executed. An agent-coded adapter remains subject to "
                "the same host validation and evidence gates as a deterministic kernel.",
                "",
                "## Validation and reporting",
                "",
                *[f"- {value}" for value in row["reporting_items"]],
                "- Verify that inputs, references, sample counts, failures, and diagnostics "
                "match the reviewed plan before interpreting the result.",
                "- Stop rather than replace this method with an unregistered alternative.",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            *[f"- {value}" for value in row["outputs"]],
            "",
            "Each output must retain its source binding, parameters, status, and digest "
            "so the result can be traced back to the exact reviewed run.",
            "",
            "## Evidence and claim boundary",
            "",
            f"- Claim ceiling: `{row['claim_ceiling']}`",
            f"- {claim_text}",
            "- Engineering success does not by itself establish scientific validity, "
            "clinical utility, or publication authority.",
            "",
            "## Failure behavior",
            "",
            "Fail closed when the source version changes, a required input is unresolved, "
            "the scientific plan is not approved, a registered action cannot run, or a "
            "required diagnostic fails. Record the failure and preserve it in the run "
            "denominator; do not silently retry with a different method or data definition.",
            "",
            "## Availability and execution",
            "",
            "This read-only package documents a capability built into EasyICU. "
            "Shared data access, plan review, method dispatch, evidence receipts, "
            "and reportability remain host-owned instead of being copied into this "
            "package. A `scripts/` directory appears only when this capability has a "
            "real package-specific helper.",
            "",
            f"## Start this {section_name}",
            "",
            f"**Task starter:** {row['prompt']}",
            "",
        ]
    )
    return "\n".join(lines)


def _package_contract_markdown(kind: str, row: dict[str, Any]) -> str:
    title = {
        "research_workflow": "Workflow contract",
        "analysis_module": "Reusable module contract",
        "method_component": "Method component contract",
    }[kind]
    lines = [
        f"# {title}: {row['title']}",
        "",
        f"- Registry id: `{row['id']}`",
        f"- Execution: `{row['execution_mode']}`",
        f"- Claim ceiling: `{row['claim_ceiling']}`",
    ]
    if kind in {"research_workflow", "analysis_module"}:
        lines.extend(
            [
                f"- Capability: `{row['capability_id']}`",
                "",
                "## Registered actions",
                "",
                *(
                    [f"- `{value}`" for value in row["action_ids"]]
                    or ["- Bound directly to the capability owner."]
                ),
                "",
                "## Inputs",
                "",
                *[f"- {value}" for value in row["inputs"]],
                "",
                "## Outputs",
                "",
                *[f"- {value}" for value in row["outputs"]],
                "",
                "## Diagnostics",
                "",
                *[f"- {value}" for value in row["diagnostics"]],
            ]
        )
    else:
        lines.extend(
            [
                f"- Method coordinate: `{row['method_family']}.{row['method_key']}`",
                f"- Tier: `{row['tier']}`",
                f"- Implementation: `{row['implementation']}`",
                "",
                "## Reviewed kernels",
                "",
                *(
                    [f"- `{value}`" for value in row["kernel_modules"]]
                    or ["- Agent-coded adapter under host gates."]
                ),
                "",
                "## Reporting bindings",
                "",
                *[f"- {value}" for value in row["reporting_items"]],
            ]
        )
    return "\n".join(lines) + "\n"


def _package_validation_markdown(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# Validation framework: {row['title']}",
            "",
            "1. Confirm every requested input and scientific definition before execution.",
            "2. Bind the reviewed plan, data authority, action ids, and software resources.",
            "3. Run through the EasyICU host and retain product digests and execution receipts.",
            "4. Check the registered diagnostics and preserve failures in the audit record.",
            f"5. Do not exceed the `{row['claim_ceiling']}` claim ceiling.",
            "",
            "The generated files are a reviewed interface to the host runtime. They do not",
            "authorize patient-data access, substitute another estimand, or bypass review.",
            "",
        ]
    )


def _package_composition_markdown(row: dict[str, Any]) -> str:
    modules = row.get("included_module_ids", [])
    return "\n".join(
        [
            f"# Workflow composition: {row['title']}",
            "",
            "The workflow may compose these reusable modules after the plan confirms",
            "their variables, time windows, denominators, and reporting rules:",
            "",
            *(
                [f"- `{module_id}`" for module_id in modules]
                or ["- No reusable module is included by default."]
            ),
            "",
            "Modules support the workflow; they do not become additional primary estimands.",
            "",
        ]
    )


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


def method_skill_package(skill_id: str, *, enabled: bool) -> dict[str, Any]:
    """Return one digest-bound, read-only package grouped by purpose."""

    kind, row, catalog_sha256 = _package_entry(skill_id, enabled=enabled)
    sources = {
        "SKILL.md": _package_skill_markdown(kind, row),
        "references/validation_framework.md": _package_validation_markdown(row),
    }
    contract_name = (
        "workflow_contract.md"
        if kind == "research_workflow"
        else "module_contract.md"
        if kind == "analysis_module"
        else "method_contract.md"
    )
    sources[f"references/{contract_name}"] = _package_contract_markdown(kind, row)
    if kind == "research_workflow":
        sources["references/composition.md"] = _package_composition_markdown(row)
    if row["id"] == "trajectory-phenotyping":
        sources["references/trajectory_assignment.md"] = _trajectory_assignment_markdown()
        sources["scripts/early_subtype_assignment.py"] = _trajectory_assignment_python()
    execution_boundary = DOCUMENTATION_ONLY_BOUNDARY
    if row["id"] in REFERENCE_SCRIPT_PACKAGES:
        sources.update(_reference_script_sources(REFERENCE_SCRIPT_PACKAGES[row["id"]]))
        execution_boundary = REFERENCE_SCRIPTS_BOUNDARY
    files = []
    for path, content in sources.items():
        encoded = content.encode("utf-8")
        files.append(
            {
                "path": path,
                "language": "markdown" if path.endswith(".md") else "python",
                "size_bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "content": content,
            }
        )
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
]
