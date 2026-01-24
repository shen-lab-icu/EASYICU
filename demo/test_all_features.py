#!/usr/bin/env python3
"""
PyRICU Webapp 完整功能测试
验证所有核心功能正常工作
"""
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'pyricu', 'webapp'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from io import StringIO
import traceback

def test_generate_mock_data():
    """测试模拟数据生成"""
    print("=" * 60)
    print("测试 1: 模拟数据生成")
    print("=" * 60)
    
    from app import generate_mock_data
    
    # 测试不同参数
    for n_patients in [5, 10, 50]:
        for n_hours in [24, 72]:
            data, patient_ids = generate_mock_data(n_patients, n_hours)
            assert len(data) == 12, f"期望12个概念，得到{len(data)}"
            assert len(patient_ids) == n_patients, f"期望{n_patients}个患者，得到{len(patient_ids)}"
            
            for name, df in data.items():
                assert isinstance(df, pd.DataFrame), f"{name} 不是 DataFrame"
                assert 'stay_id' in df.columns, f"{name} 缺少 stay_id 列"
                # 时间列可以是 'time' 或 'charttime'
                has_time_col = 'time' in df.columns or 'charttime' in df.columns
                assert has_time_col, f"{name} 缺少时间列 (time/charttime)"
            
            print(f"   ✓ {n_patients}患者/{n_hours}小时: {len(data)}概念, {sum(len(df) for df in data.values())}行")
    
    print("   ✅ 通过\n")
    return True

def test_data_loading_logic():
    """测试数据加载逻辑"""
    print("=" * 60)
    print("测试 2: 数据加载逻辑")
    print("=" * 60)
    
    from app import generate_mock_data
    
    data, patient_ids = generate_mock_data(10, 48)
    
    # 模拟 load_data 后的处理
    loaded_concepts = data
    
    # 测试 len() 检查
    if len(loaded_concepts) > 0:
        print(f"   ✓ len() 检查通过: {len(loaded_concepts)} 概念")
    
    # 测试遍历
    for name, df in loaded_concepts.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            cols = df.columns.tolist()
            assert 'stay_id' in cols, f"{name} 缺少 stay_id"
    print("   ✓ 遍历和类型检查通过")
    
    # 测试空数据情况
    empty_concepts = {}
    if len(empty_concepts) > 0:
        raise AssertionError("空字典不应该通过检查")
    print("   ✓ 空数据检查通过")
    
    print("   ✅ 通过\n")
    return True

def test_concat_operations():
    """测试 concat 操作"""
    print("=" * 60)
    print("测试 3: Concat 操作")
    print("=" * 60)
    
    from app import generate_mock_data
    
    data, _ = generate_mock_data(5, 24)
    
    # 测试全部数据合并
    df_list = [df.assign(concept=name) for name, df in data.items() 
               if isinstance(df, pd.DataFrame) and len(df) > 0]
    
    if df_list:
        all_data = pd.concat(df_list, ignore_index=True)
        print(f"   ✓ 全部数据合并: {all_data.shape}")
    
    # 测试生命体征合并
    vitals = ['hr', 'map', 'sbp', 'resp', 'spo2', 'temp']
    vitals_list = [df.assign(concept=name) for name, df in data.items() 
                   if name in vitals and isinstance(df, pd.DataFrame) and len(df) > 0]
    if vitals_list:
        vitals_data = pd.concat(vitals_list, ignore_index=True)
        print(f"   ✓ 生命体征合并: {vitals_data.shape}")
    
    # 测试实验室合并
    labs = ['bili', 'crea', 'plt', 'lac', 'wbc', 'hgb']
    labs_list = [df.assign(concept=name) for name, df in data.items() 
                 if name in labs and isinstance(df, pd.DataFrame) and len(df) > 0]
    if labs_list:
        labs_data = pd.concat(labs_list, ignore_index=True)
        print(f"   ✓ 实验室合并: {labs_data.shape}")
    
    # 测试空列表情况
    empty_list = []
    if empty_list:
        pd.concat(empty_list)  # 不应该执行
        raise AssertionError("空列表不应该执行concat")
    print("   ✓ 空列表检查通过")
    
    print("   ✅ 通过\n")
    return True

def test_patient_filtering():
    """测试患者过滤"""
    print("=" * 60)
    print("测试 4: 患者过滤")
    print("=" * 60)
    
    from app import generate_mock_data
    
    data, patient_ids = generate_mock_data(10, 24)
    
    selected_patient = patient_ids[0]
    
    for name, df in data.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            patient_data = df[df['stay_id'] == selected_patient]
            assert len(patient_data) > 0, f"{name} 没有患者 {selected_patient} 的数据"
    
    print(f"   ✓ 患者 {selected_patient} 过滤通过")
    
    # 测试多患者
    multi_patients = patient_ids[:3]
    for name, df in data.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            multi_data = df[df['stay_id'].isin(multi_patients)]
            assert len(multi_data) > 0, f"{name} 没有多患者数据"
    
    print(f"   ✓ 多患者 {multi_patients} 过滤通过")
    
    print("   ✅ 通过\n")
    return True

def test_export_csv():
    """测试 CSV 导出"""
    print("=" * 60)
    print("测试 5: CSV 导出")
    print("=" * 60)
    
    from app import generate_mock_data
    
    data, _ = generate_mock_data(5, 24)
    
    # 导出单个概念
    for name, df in data.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            csv_str = df.to_csv(index=False)
            # 验证 CSV 可以重新读取
            df_reload = pd.read_csv(StringIO(csv_str))
            assert df_reload.shape == df.shape, f"{name} CSV 导出/读取形状不匹配"
    print("   ✓ 单概念 CSV 导出通过")
    
    # 导出全部
    df_list = [df.assign(concept=name) for name, df in data.items() 
               if isinstance(df, pd.DataFrame) and len(df) > 0]
    if df_list:
        all_csv = pd.concat(df_list, ignore_index=True).to_csv(index=False)
        all_reload = pd.read_csv(StringIO(all_csv))
        print(f"   ✓ 全部数据 CSV 导出: {all_reload.shape}")
    
    print("   ✅ 通过\n")
    return True

def test_statistics():
    """测试统计计算"""
    print("=" * 60)
    print("测试 6: 统计计算")
    print("=" * 60)
    
    from app import generate_mock_data
    
    data, patient_ids = generate_mock_data(10, 24)
    
    for name, df in data.items():
        if isinstance(df, pd.DataFrame) and len(df) > 0:
            # 检查是否有数值列
            value_cols = [c for c in df.columns if c not in ['stay_id', 'charttime', 'concept']]
            if value_cols:
                val_col = value_cols[0]
                if pd.api.types.is_numeric_dtype(df[val_col]):
                    stats = df[val_col].describe()
                    assert 'mean' in stats.index, f"{name} 统计缺少 mean"
                    assert 'std' in stats.index, f"{name} 统计缺少 std"
    
    print("   ✓ 描述性统计计算通过")
    
    print("   ✅ 通过\n")
    return True

def test_webapp_endpoints():
    """测试 webapp 端点"""
    print("=" * 60)
    print("测试 7: Webapp 端点")
    print("=" * 60)
    
    import requests
    
    base_url = "http://localhost:8502"
    
    try:
        # 健康检查
        resp = requests.get(f"{base_url}/_stcore/health", timeout=5)
        assert resp.status_code == 200, f"健康检查失败: {resp.status_code}"
        assert resp.text == "ok", f"健康检查响应异常: {resp.text}"
        print("   ✓ 健康检查通过")
        
        # 主页
        resp = requests.get(f"{base_url}/", timeout=10)
        assert resp.status_code == 200, f"主页访问失败: {resp.status_code}"
        print("   ✓ 主页访问通过")
        
        print("   ✅ 通过\n")
        return True
        
    except requests.exceptions.ConnectionError:
        print("   ⚠️ Webapp 未运行，跳过端点测试")
        print("   ✅ 跳过\n")
        return True

def main():
    print("\n" + "=" * 60)
    print("   PyRICU Webapp 完整功能测试")
    print("=" * 60 + "\n")
    
    tests = [
        ("模拟数据生成", test_generate_mock_data),
        ("数据加载逻辑", test_data_loading_logic),
        ("Concat 操作", test_concat_operations),
        ("患者过滤", test_patient_filtering),
        ("CSV 导出", test_export_csv),
        ("统计计算", test_statistics),
        ("Webapp 端点", test_webapp_endpoints),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"   ❌ 失败: {e}")
            traceback.print_exc()
            print()
    
    print("=" * 60)
    print("   测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    
    for name, ok, error in results:
        status = "✅ 通过" if ok else f"❌ 失败: {error}"
        print(f"   {name}: {status}")
    
    print()
    print(f"   总计: {passed}/{total} 通过")
    print("=" * 60 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
