#!/usr/bin/env python3
"""测试 PyRICU Webapp 核心功能。"""

import sys
sys.path.insert(0, '/home/zhuhb/project/ricu_to_python/pyricu/src')

import pandas as pd
import numpy as np

def test_generate_mock_data():
    """测试模拟数据生成。"""
    print("测试 1: 模拟数据生成...")
    
    from pyricu.webapp.app import generate_mock_data
    
    data, patient_ids = generate_mock_data(n_patients=5, hours=24)
    
    assert isinstance(data, dict), f"Expected dict, got {type(data)}"
    assert len(data) > 0, "Data should not be empty"
    assert len(patient_ids) == 5, f"Expected 5 patients, got {len(patient_ids)}"
    
    for name, df in data.items():
        assert isinstance(df, pd.DataFrame), f"{name} should be DataFrame, got {type(df)}"
        assert len(df) > 0, f"{name} should not be empty"
        assert 'stay_id' in df.columns, f"{name} should have stay_id column"
        assert 'time' in df.columns, f"{name} should have time column"
    
    print(f"  ✓ 生成了 {len(data)} 个 concepts")
    print(f"  ✓ 每个 concept 都是 DataFrame")
    return data, patient_ids


def test_concat_logic(data):
    """测试 concat 逻辑。"""
    print("\n测试 2: concat 逻辑...")
    
    # 测试正常情况
    df_list = [df.assign(concept=name) for name, df in data.items() 
               if isinstance(df, pd.DataFrame) and len(df) > 0]
    
    assert len(df_list) > 0, "df_list should not be empty"
    
    all_data = pd.concat(df_list, ignore_index=True)
    assert isinstance(all_data, pd.DataFrame), "concat result should be DataFrame"
    assert len(all_data) > 0, "concat result should not be empty"
    
    print(f"  ✓ df_list 长度: {len(df_list)}")
    print(f"  ✓ concat 结果形状: {all_data.shape}")
    
    # 测试空字典
    empty_data = {}
    df_list2 = [df.assign(concept=name) for name, df in empty_data.items() 
                if isinstance(df, pd.DataFrame) and len(df) > 0]
    assert len(df_list2) == 0, "Empty dict should produce empty list"
    print("  ✓ 空字典正确处理")
    
    return all_data


def test_len_checks(data):
    """测试 len() 检查逻辑。"""
    print("\n测试 3: len() 检查...")
    
    # 测试空字典
    empty = {}
    assert len(empty) == 0, "Empty dict len should be 0"
    
    # 测试有数据的字典
    assert len(data) > 0, "Data dict len should be > 0"
    
    print("  ✓ len() 检查正常")


def test_isinstance_checks(data):
    """测试 isinstance 检查。"""
    print("\n测试 4: isinstance 检查...")
    
    for name, df in data.items():
        assert isinstance(df, pd.DataFrame), f"{name} isinstance check failed"
        
        # 确保不是 Series
        assert not isinstance(df, pd.Series), f"{name} should not be Series"
    
    print("  ✓ 所有数据都是 DataFrame")


def test_export_logic(data):
    """测试导出逻辑。"""
    print("\n测试 5: 导出逻辑...")
    
    # 生命体征筛选
    vitals = ['hr', 'map', 'sbp', 'resp', 'spo2', 'temp']
    vitals_data = {k: v for k, v in data.items() 
                  if k in vitals and isinstance(v, pd.DataFrame) and len(v) > 0}
    
    if vitals_data:
        vitals_list = [df.assign(concept=name) for name, df in vitals_data.items()]
        vitals_combined = pd.concat(vitals_list, ignore_index=True)
        vitals_csv = vitals_combined.to_csv(index=False)
        assert len(vitals_csv) > 0, "CSV should not be empty"
        print(f"  ✓ 生命体征导出成功: {len(vitals_csv)} 字符")
    else:
        print("  ✓ 无生命体征数据，正确跳过")


def main():
    print("=" * 50)
    print("PyRICU Webapp 功能测试")
    print("=" * 50)
    
    try:
        data, patient_ids = test_generate_mock_data()
        test_concat_logic(data)
        test_len_checks(data)
        test_isinstance_checks(data)
        test_export_logic(data)
        
        print("\n" + "=" * 50)
        print("✅ 所有测试通过!")
        print("=" * 50)
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
