#!/usr/bin/env python3
"""验证pyricu功能实现状态的脚本.

读取feature_tracking.json并验证实际的代码实现情况。
"""

import json
import os
import sys
from pathlib import Path

# 添加pyricu到路径
SCRIPT_DIR = Path(__file__).parent
PYRICU_SRC = SCRIPT_DIR / "src" / "pyricu"
sys.path.insert(0, str(SCRIPT_DIR / "src"))


def load_tracking_json():
    """加载功能追踪JSON文件."""
    tracking_file = SCRIPT_DIR / "feature_tracking.json"
    with open(tracking_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_concept_dicts():
    """加载概念字典文件."""
    data_dir = PYRICU_SRC / "data"
    
    concepts = {}
    
    # 主概念字典
    concept_dict_path = data_dir / "concept-dict.json"
    if concept_dict_path.exists():
        with open(concept_dict_path, "r", encoding="utf-8") as f:
            concepts["main"] = json.load(f)
    
    # SOFA2字典
    sofa2_dict_path = data_dir / "sofa2-dict.json"
    if sofa2_dict_path.exists():
        with open(sofa2_dict_path, "r", encoding="utf-8") as f:
            concepts["sofa2"] = json.load(f)
    
    return concepts


def check_callbacks_exist():
    """检查回调函数是否存在于代码中."""
    callbacks_to_check = [
        "sofa_score", "sofa_resp", "sofa_coag", "sofa_liver", 
        "sofa_cardio", "sofa_cns", "sofa_renal",
        "sofa2_score", "sofa2_resp", "sofa2_coag", "sofa2_liver",
        "sofa2_cardio", "sofa2_cns", "sofa2_renal",
        "sirs_score", "qsofa_score", "news_score", "mews_score",
        "sep3", "susp_inf",
        "pafi", "safi", "gcs", "vent_ind", "vaso_ind", "urine24",
        "norepi_equiv", "bmi", "avpu", "supp_o2",
        "fill_gaps", "slide", "expand", "collapse", "hop",
    ]
    
    found = {}
    not_found = []
    
    # 搜索Python文件
    for py_file in PYRICU_SRC.glob("*.py"):
        content = py_file.read_text()
        for callback in callbacks_to_check:
            if f"def {callback}(" in content:
                found[callback] = py_file.name
    
    for callback in callbacks_to_check:
        if callback not in found:
            not_found.append(callback)
    
    return found, not_found


def check_table_classes():
    """检查表类是否已实现."""
    classes = ["IdTbl", "TsTbl", "WinTbl", "ICUTable"]
    found = {}
    
    table_file = PYRICU_SRC / "table.py"
    if table_file.exists():
        content = table_file.read_text()
        for cls in classes:
            if f"class {cls}" in content:
                found[cls] = True
    
    return found


def validate_concepts():
    """验证概念定义."""
    concepts = load_concept_dicts()
    
    results = {
        "main_concepts_count": len(concepts.get("main", {})),
        "sofa2_concepts_count": len(concepts.get("sofa2", {})),
        "main_concepts": list(concepts.get("main", {}).keys()),
        "sofa2_concepts": list(concepts.get("sofa2", {}).keys()),
    }
    
    return results


def generate_report():
    """生成验证报告."""
    print("=" * 60)
    print("PYRICU 功能实现验证报告")
    print("=" * 60)
    
    # 1. 加载追踪文件
    print("\n1. 加载功能追踪文件...")
    tracking = load_tracking_json()
    print(f"   ✓ 成功加载 feature_tracking.json")
    
    # 2. 验证概念定义
    print("\n2. 验证概念定义...")
    concept_results = validate_concepts()
    print(f"   ✓ 主概念字典: {concept_results['main_concepts_count']} 个概念")
    print(f"   ✓ SOFA2字典: {concept_results['sofa2_concepts_count']} 个概念")
    
    # 3. 检查回调函数
    print("\n3. 检查回调函数实现...")
    found_callbacks, missing_callbacks = check_callbacks_exist()
    print(f"   ✓ 已找到: {len(found_callbacks)} 个回调函数")
    if missing_callbacks:
        print(f"   ! 未找到: {missing_callbacks}")
    else:
        print(f"   ✓ 所有核心回调函数已实现")
    
    # 4. 检查表类
    print("\n4. 检查数据表类...")
    table_classes = check_table_classes()
    for cls, exists in table_classes.items():
        status = "✓" if exists else "✗"
        print(f"   {status} {cls}")
    
    # 5. 统计功能覆盖
    print("\n5. 功能覆盖统计:")
    
    # 从追踪文件提取统计
    scores = tracking.get("scores", {}).get("implemented", {})
    print(f"   ✓ 临床评分: {len(scores)} 个")
    
    concept_callbacks = tracking.get("concept_callbacks", {}).get("implemented", {})
    print(f"   ✓ 概念回调: {len(concept_callbacks)} 个")
    
    ts_utils = tracking.get("time_series_utilities", {}).get("implemented", {})
    print(f"   ✓ 时间序列工具: {len(ts_utils)} 个")
    
    data_sources = tracking.get("data_sources", {})
    full_support = len(data_sources.get("fully_supported", {}))
    partial_support = len(data_sources.get("partially_supported", {}))
    print(f"   ✓ 数据库支持: {full_support} 完全支持, {partial_support} 部分支持")
    
    # 6. SOFA2新增功能
    print("\n6. SOFA-2 (2025) 新增功能:")
    sofa2_exclusive = tracking.get("pyricu_exclusive_features", {}).get("sofa2_system", {})
    for component in sofa2_exclusive.get("components", []):
        print(f"   ✓ {component}")
    
    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)
    
    return {
        "concepts": concept_results,
        "callbacks_found": len(found_callbacks),
        "callbacks_missing": missing_callbacks,
        "table_classes": table_classes,
    }


def main():
    """主函数."""
    try:
        results = generate_report()
        
        # 保存结果
        output_file = SCRIPT_DIR / "validation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n验证结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
