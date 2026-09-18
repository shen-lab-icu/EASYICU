"""Export a read-only, public-field inventory from the shipped EasyICU registries."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from easyicu.concept import catalog as C  # noqa: E402
from easyicu.research_agent.planning.capability_registry import (  # noqa: E402
    CAPABILITY_REGISTRY,
)

MODULE_LABELS = {
    "sofa2_score": "SOFA-2 评分", "sofa1_score": "SOFA-1 评分",
    "sepsis3_sofa2": "脓毒症 · SOFA-2", "sepsis3_sofa1": "脓毒症 · SOFA-1",
    "sepsis_shared": "感染与共同定义", "vitals": "生命体征", "respiratory": "呼吸支持",
    "ventilator": "机械通气", "blood_gas": "血气分析", "chemistry": "生化检验",
    "hematology": "血液学", "vasopressors": "血管活性药物", "medications": "药物治疗",
    "renal": "肾脏与尿量", "neurological": "神经系统", "circulatory": "循环系统",
    "demographics": "人口学", "other_scores": "其他评分", "outcome": "研究结局",
}
METHOD_COPY = {
    "association_time_varying_exposure_v1": ("时变暴露关联", "按明确的时间更新规则构建 Cox 关联分析。", "开发中的分析能力；需确认区间、时序及模型诊断。"),
    "survival_time_to_event_v1": ("生存与时间结局", "按审阅方案执行 Cox 模型，组织随访、事件和诊断结果。", "限定于声明的模型合同；需要满足时间起点、删失与比例风险要求。"),
    "causal_target_trial_v1": ("因果推断与目标试验模拟", "在声明的识别策略下生成分析代码，组织平衡性与效应结果。", "探索性分析；不能将代码执行成功等同于因果识别成立。"),
    "source_feasibility_non_use_v1": ("因果问题的数据可行性", "检查数据是否能区分所需的治疗使用与未使用状态。", "可给出有据可查的不可识别结论，不计算或证明治疗效果。"),
    "association_ordinal_trend_v1": ("有序暴露与趋势", "处理分级暴露及剂量反应问题，保留类别次序。", "分析级能力；具体模型和调整集由研究方案确定。"),
    "association_adjusted_v1": ("调整后关联分析", "执行明确暴露、结局与协变量的单模型关联分析。", "限定模型设定的结果核验，不代表消除所有混杂。"),
    "association_landmark_categorical_v1": ("Landmark 分类暴露", "对齐时间窗与入组条件，分析分类暴露与后续结局。", "需要声明 landmark、分组、结局和模型设置。"),
    "association_landmark_spline_v1": ("Landmark 样条关联", "建模连续暴露的非线性关联，并输出函数形式比较与图表。", "已有乳酸开发案例；其他研究仍需逐题检查数据和假设。"),
    "association_freeform_v1": ("自由形式关联分析", "由研究方案组织交互、样条和多模型等分析代码。", "探索性代码执行路径；不能复用其他方法的核验结论。"),
    "prediction_risk_model_v1": ("静态风险预测", "围绕固定的 L2 Logistic 模型组织拟合与预测评价。", "需检查患者级划分、泄漏和校准；不等于临床可用模型。"),
    "dynamic_prediction_landmark_v1": ("动态预测与预警", "按 landmark 组织动态预测问题及评价输入。", "模型拟合仍依赖生成代码，目前保持分析级范围。"),
    "phenotyping_cluster_v1": ("探索性聚类分型", "组织特征、缺失处理、标准化、聚类与稳定性评价。", "用于探索候选分组，不能直接命名为已确立的临床亚型。"),
    "trajectory_signed_phenotyping_v1": ("固定窗口轨迹分型", "在明确的时间锚点与窗口内组织轨迹聚类。", "依赖完整审阅的输入与配置，不授权因果或生物学分型结论。"),
    "descriptive_measurement_v1": ("描述分析与测量审计", "检查变量分布、缺失与测量过程，描述研究样本。", "广义分析流程，具体产物需结合研究计划审阅。"),
    "descriptive_exposure_outcome_distribution_v1": ("暴露分布与绝对风险", "按明确分母计算暴露分布与结局比例。", "限定已声明人群与结局定义的描述结果。"),
}

def public_source_url(url):
    """Link visitors to dataset descriptions, rather than credentialed file roots."""
    if url.startswith("https://physionet.org/files/"):
        return url.replace("https://physionet.org/files/", "https://physionet.org/content/", 1).rstrip("/") + "/"
    return url

def main():
    paths = [ROOT / "src/easyicu/data" / name for name in ("concept-dict.json", "sofa2-dict.json", "data-sources.json")]
    base, overlay, sources = [json.loads(p.read_text()) for p in paths]
    sources = [source for source in sources if not source["name"].endswith("_demo")]
    merged = {**base, **overlay}
    source_ids = {s["name"] for s in sources}
    groups = C.CONCEPT_GROUPS_INTERNAL
    concepts = []
    for key, value in sorted(merged.items()):
        meta = C.CONCEPT_DICTIONARY.get(key, (value.get("description", key), "", ""))
        description = C.CONCEPT_DESCRIPTIONS.get(key, (value.get("description", ""), ""))
        concepts.append({
            "id": key, "name": meta[1] or meta[0], "english": meta[0], "unit": meta[2],
            "description": description[1] or description[0],
            "modules": [module for module, keys in groups.items() if key in keys],
            "directSources": sorted(set(value.get("sources", {})) & source_ids),
            "derived": bool(value.get("callback") or value.get("sub_concepts")),
        })
    methods = []
    for capability in CAPABILITY_REGISTRY:
        name, description, boundary = METHOD_COPY[capability.capability_id]
        methods.append({"id": capability.capability_id, "name": name, "english": capability.label,
                        "family": capability.family, "description": description, "boundary": boundary,
                        "path": "固定实现" if capability.primary_analysis == "deterministic" else "生成代码",
                        "validation": "已登记结果核验" if capability.scientific_validation == "reportable" else "分析级能力"})
    data = {"updated": str(date.today()), "counts": {"concepts": len(concepts), "sources": len(sources), "modules": len(groups), "methods": len(methods)},
            "modules": [{"id": k, "name": MODULE_LABELS[k]} for k in groups],
            "sources": [{"id": s["name"], "name": s.get("profile", {}).get("display_name", s["name"]), "release": s.get("profile", {}).get("reference_release", ""), "url": public_source_url(s.get("url", ""))} for s in sorted(sources, key=lambda s:s.get("profile", {}).get("display_order", 100))],
            "concepts": concepts, "methods": methods}
    out = ROOT / "website/dist/catalog-data.js"
    out.write_text("window.EASYICU_CATALOG = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    paths.extend([Path(C.__file__), ROOT / "src/easyicu/research_agent/planning/capability_registry.py"])
    provenance = {"counts":data["counts"],"inputs":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT/"website/catalog-provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps(data["counts"]))

if __name__ == "__main__":
    main()
