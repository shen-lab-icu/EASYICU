"""Public, method-first Skill Hub projection of real EasyICU capabilities.

The research-agent registries own execution and scientific authority.  This
module owns only the user-facing catalogue: a small set of discoverable method
workflows whose capability and action identifiers are checked at import time.
It deliberately exposes no Python entrypoints or implementation paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Literal, Tuple

from .planning.capability_registry import CAPABILITY_REGISTRY
from .planning.analysis_method_suite import METHOD_SUITE_REGISTRY
from .planning.method_adapter_catalog import HIGH_FREQUENCY_METHOD_ADAPTERS

METHOD_SKILL_REGISTRY_VERSION = "easyicu.method-skills/1"

ClaimCeiling = Literal["reportable", "analysis_only"]
ExecutionMode = Literal["deterministic_host", "agent_coded_with_host_gates"]


@dataclass(frozen=True, slots=True)
class MethodSkillSpec:
    """One discoverable research-method workflow backed by registered owners."""

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
        "Use the cohort-characterization and Table 1 workflow. First confirm the cohort, grouping variable, closed grouping levels, variables, units, and summary rules. Then build the typed plan and run only after the scientific plan is reviewed.",
        "请使用“队列描述与 Table 1”方法。先确认研究队列、分组变量与封闭分组水平、待汇总变量、单位和汇总规则，再生成类型明确的方法计划；科学计划审阅通过后再运行。",
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
        "Use the missingness and measurement audit workflow. Confirm the cohort, audited variables, time windows, denominators, structural absence rules, and requested missingness products before planning execution.",
        "请使用“缺失与测量审计”方法。先确认队列、审计变量、时间窗、分母、结构性缺失规则和需要输出的缺失数据产物，再制定执行计划。",
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
        "Use the exposure and outcome distribution workflow. Confirm the cohort, closed exposure levels, outcome definition, denominator, interval method, and dependence structure. Keep the interpretation descriptive and non-causal.",
        "请使用“研究因素与结局分布”方法。先确认队列、封闭的研究因素水平、结局定义、分母、区间方法和相关性结构；解释必须保持描述性，不作因果推断。",
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
        "Use the adjusted association workflow. Confirm one exposure, one outcome, estimator, covariate roster, coding, reference levels, transforms, population, and missing-data policy before producing the model contract.",
        "请使用“调整后关联模型”方法。先确认一个研究因素、一个结局、估计量、协变量清单、编码方式、参考水平、变量变换、分析人群和缺失数据策略，再生成模型契约。",
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
        "Use the ordinal trend and dose-response workflow. Confirm at least three ordered levels, their order and reference, the primary contrast, outcome, adjustment set, and trend method. Do not coerce a binary or continuous exposure into ordinal levels.",
        "请使用“有序趋势与剂量反应”方法。先确认至少三个有序水平及其顺序与参考水平、主要对比、结局、调整集和趋势检验方法；不得把二分类或连续研究因素强行改成有序分组。",
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
        "Use the fixed-landmark categorical association workflow. Confirm time zero, landmark, observation opportunity, eligible population, exposure levels and reference, binary outcome, adjustment set, and dependence contract.",
        "请使用“Landmark 分类关联”方法。先确认时间零点、landmark、观察机会、合格人群、研究因素水平与参考水平、二分类结局、调整集和相关性契约。",
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
        "Use the fixed-landmark spline workflow. Confirm time zero, landmark, exposure window, continuous exposure, knots, reference value, outcome, adjustment set, and the spline-versus-linear comparison before planning execution.",
        "请使用“Landmark 样条剂量反应”方法。先确认时间零点、landmark、研究因素窗口、连续研究因素、结点、参考值、结局、调整集以及样条与线性模型的比较方式，再制定执行计划。",
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
        "Use the time-varying exposure Cox workflow. Confirm time zero, interval construction, exposure update rule, unmeasured-state handling, event and censoring rules, baseline coding, covariates, and patient clustering.",
        "请使用“时间变化研究因素 Cox 模型”方法。先确认时间零点、区间构造、研究因素更新规则、未测量状态处理、事件与删失规则、基线编码、协变量以及患者聚类。",
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
        "Use the survival and time-to-event workflow. Confirm time origin and unit, event indicator and value, censoring, exposure and reference, exact covariates, horizon, complete-case policy, and proportional-hazards policy.",
        "请使用“生存与时间结局分析”方法。先确认时间起点与单位、事件指示及取值、删失规则、研究因素与参考水平、精确协变量、观察期限、完整案例策略和比例风险检验策略。",
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
        "Use the RMST workflow. Confirm the two groups and reference, time origin and unit, event and censoring definitions, analysis horizon, population, and interval method before execution.",
        "请使用“限制性平均生存时间（RMST）”方法。先确认两组及参考组、时间起点与单位、事件与删失定义、分析时间范围、人群和区间方法，再执行分析。",
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
        "Use the clinical risk prediction workflow. Confirm the prediction time, target horizon, binary outcome, predictors available at prediction time, patient-level split, preprocessing, internal validation, calibration, and clinical-utility plan.",
        "请使用“临床风险预测”方法。先确认预测时点、目标时间范围、二分类结局、预测时点可用的预测变量、患者级数据划分、预处理、内部验证、校准和临床效用评估方案。",
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
        "Use the dynamic landmark prediction workflow. Confirm patient identity, measurement times, landmarks, lookback windows, target horizons, censoring and observability, predictors, and patient-level development and validation splits.",
        "请使用“动态 Landmark 预测”方法。先确认患者标识、测量时间、landmark、回看窗口、目标时间范围、删失与可观察性、预测变量，以及患者级开发集和验证集划分。",
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
        "Use the target-trial emulation workflow. First specify eligibility, treatment strategies, assignment, time zero, follow-up, outcome, causal contrast, censoring, confounding strategy, positivity checks, and falsification analyses. Keep the result analysis-only unless independent identification and readiness gates pass.",
        "请使用“目标试验模拟”方法。先明确纳入标准、治疗策略、分配方式、时间零点、随访、结局、因果对比、删失、混杂控制、可交换性/阳性检查和证伪分析；除非独立的识别与就绪门禁通过，否则结果保持“仅分析”。",
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
        "Use the cross-sectional phenotyping workflow. Confirm the cohort, feature window, feature roster excluding outcomes, scaling and missingness rules, closed candidate-k grid, selection rule, stability design, and descriptive outcome comparison.",
        "请使用“横断面表型聚类”方法。先确认队列、特征窗口、不包含结局的特征清单、标准化与缺失规则、封闭的候选 k 网格、选择规则、稳定性设计和描述性结局比较。",
    ),
    MethodSkillSpec(
        "trajectory-phenotyping",
        "Trajectory phenotyping",
        "轨迹表型聚类",
        "Phenotyping",
        "表型识别",
        "Build fixed-window trajectory phenotypes with closed selection and stability rules.",
        "使用封闭的选择与稳定性规则构建固定窗口轨迹表型。",
        "trajectory_signed_phenotyping_v1",
        ("phenotyping.trajectory_cluster_stability",),
        "deterministic_host",
        "reportable",
        "Use the trajectory phenotyping workflow. Confirm the fixed time window, trajectory representation, variables and missingness rules, closed candidate grid, BIC selection rule, resampling and alignment policy, stability threshold, and descriptive outcome use.",
        "请使用“轨迹表型聚类”方法。先确认固定时间窗、轨迹表示、变量与缺失规则、封闭候选网格、BIC 选择规则、重采样与标签对齐策略、稳定性阈值及结局的描述性用途。",
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
    "causal_emulation.iptw_or": ("稳定化 IPTW 因果对比", "在目标试验方案下估计边际因果对比。"),
    "causal_emulation.covariate_balance": ("协变量平衡与 Love plot", "用加权前后的标准化差异检查已测混杂平衡。"),
    "causal_emulation.positivity_overlap": ("阳性与重叠诊断", "检查治疗组之间的倾向评分重叠和极端权重。"),
    "causal_emulation.evalue": ("E-value 未测混杂敏感性", "量化未测混杂需要多强才能解释观察到的关联。"),
    "causal_emulation.negative_control": ("阴性对照分析", "使用预设阴性对照结局或研究因素检查残余偏倚。"),
    "association.ordinal_trend": ("有序趋势分析", "评估封闭有序水平之间的剂量反应与单调趋势。"),
    "association.adjusted_association": ("调整后关联模型", "拟合参考水平、变量类型和调整集均已声明的关联模型。"),
    "association.multiple_adjustment": ("多层调整模型", "比较粗模型、最小调整和完整调整方案。"),
    "association.effect_modification": ("效应修饰与交互作用", "检查预设修饰因素对应的交互项与亚组结果。"),
    "association.missingness_audit": ("缺失数据与完整案例敏感性", "比较缺失模式、完整案例与预先声明的处理策略。"),
    "association.multiple_testing": ("多重检验校正", "使用 FDR 或 Bonferroni 等预设规则控制多重比较。"),
    "association.evalue": ("E-value 关联敏感性", "评估未测混杂解释调整后关联所需的最低强度。"),
    "association.robustness_panel": ("稳健性规格面板", "汇总不同队列或模型规格下主要估计的稳定性。"),
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
    "phenotyping.cluster_solution": ("横断面亚表型聚类", "基于早期特征形成并检查患者亚表型。"),
    "phenotyping.k_selection": ("聚类数量选择", "使用 silhouette、gap 或 BIC 等预设标准选择聚类数。"),
    "phenotyping.cluster_stability": ("聚类稳定性与复现性", "通过重采样、共识或调整 Rand 指数检查稳定性。"),
    "phenotyping.trajectory_cluster_stability": ("轨迹聚类稳定性复拟合", "按登记方案重采样并复核已选轨迹模型的稳定性。"),
    "phenotyping.cluster_sizes": ("聚类规模与退化检查", "报告各聚类规模并标记近空或退化聚类。"),
    "phenotyping.outcome_by_cluster": ("聚类间结局描述", "描述冻结聚类之间的临床结局差异，不作因果解释。"),
    "phenotyping.trajectory_feature_clustering": ("轨迹特征聚类", "根据预设时间窗和表示方法构建纵向轨迹表型。"),
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
                        f"Use {method.name} as the requested method component. "
                        "First confirm applicability, inputs, assumptions, comparison, and output contract. "
                        "Keep the component analysis-only unless the enclosing reviewed workflow grants a higher ceiling."
                    ),
                    "prompt_zh": (
                        f"请在研究方案中使用“{title_zh}”方法组件。先确认适用性、输入、假设、比较方式和输出契约；"
                        "除非外层经过审阅的研究工作流明确授予更高权限，否则保持“仅分析”。"
                    ),
                }
            )
    return items, planned_count


def _validate_method_skills() -> None:
    ids = [row.skill_id for row in METHOD_SKILLS]
    if len(ids) != len(set(ids)):
        raise RuntimeError("method Skill ids must be unique")
    for row in METHOD_SKILLS:
        capability = _CAPABILITY_BY_ID.get(row.capability_id)
        if capability is None:
            raise RuntimeError(f"unknown method Skill capability: {row.capability_id}")
        unknown_actions = set(row.action_ids) - _ACTION_IDS
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
    return {
        "version": METHOD_SKILL_REGISTRY_VERSION,
        "enabled": bool(enabled),
        "status": "enabled" if enabled else "disabled",
        "catalog_sha256": hashlib.sha256(canonical).hexdigest(),
        "items": items,
        "components": components,
        "workflow_count": len(items),
        "available_method_count": len(components),
        "planned_method_count": planned_count,
        "active_skill_ids": [row["id"] for row in items if row["enabled"]],
        "active_component_ids": [
            row["id"] for row in components if row["enabled"]
        ],
    }


__all__ = [
    "METHOD_SKILL_REGISTRY_VERSION",
    "METHOD_SKILLS",
    "MethodSkillSpec",
    "method_skill_catalog",
]
