#!/usr/bin/env python3
"""验证所有生成的测试数据是否可以正常加载"""

import sys
import pandas as pd
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent

# 数据库配置
DATABASES = {
    'miiv': {
        'test_dir': ROOT_DIR / 'test_data_miiv',
        'id_file': 'test_patient_ids.py',
        'id_vars': ['SELECTED_STAY_IDS', 'SELECTED_SUBJECT_IDS'],
        'required_files': [
            'icustays.parquet', 'patients.parquet', 'chartevents.parquet',
            'labevents.parquet', 'inputevents.parquet', 'outputevents.parquet'
        ]
    },
    'aumc': {
        'test_dir': ROOT_DIR / 'test_data_aumc',
        'id_file': 'test_patient_ids.py',
        'id_vars': ['SELECTED_ADMISSION_IDS'],
        'required_files': [
            'admissions.parquet', 'numericitems.parquet', 'listitems.parquet',
            'procedureorderitems.parquet', 'drugitems.parquet'
        ]
    },
    'eicu': {
        'test_dir': ROOT_DIR / 'test_data_eicu',
        'id_file': 'test_patient_ids.py',
        'id_vars': ['SELECTED_PATIENT_IDS'],
        'required_files': [
            'patient.parquet', 'vitalPeriodic.parquet', 'lab.parquet',
            'infusiondrug.parquet', 'nurseCharting.parquet', 'treatment.parquet'
        ]
    },
    'hirid': {
        'test_dir': ROOT_DIR / 'test_data_hirid',
        'id_file': 'test_patient_ids.py',
        'id_vars': ['SELECTED_PATIENT_IDS'],
        'required_files': [
            'general.parquet', 'variables.parquet', 'ordinal.parquet',
            'observations/part-00.parquet', 'pharma/part-00.parquet'
        ]
    }
}

def check_file_exists(file_path, description):
    """检查文件是否存在"""
    if file_path.exists():
        size_mb = file_path.stat().st_size / (1024 * 1024)
        return True, size_mb
    else:
        return False, 0

def verify_parquet_file(file_path, description):
    """验证parquet文件是否可以正常读取"""
    try:
        df = pd.read_parquet(file_path)
        return True, len(df), len(df.columns)
    except Exception as e:
        return False, 0, str(e)

def verify_patient_ids(test_dir, id_file, id_vars):
    """验证患者ID文件"""
    try:
        # 动态导入患者ID
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_patient_ids", test_dir / id_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ids_info = {}
        for var in id_vars:
            if hasattr(module, var):
                ids_info[var] = getattr(module, var)
            else:
                ids_info[var] = None

        return True, ids_info
    except Exception as e:
        return False, str(e)

def verify_database(db_name, config):
    """验证单个数据库的数据"""
    print(f"\n{'='*50}")
    print(f"🔍 验证 {db_name.upper()} 数据库")
    print(f"{'='*50}")

    test_dir = config['test_dir']
    results = {
        'exists': False,
        'id_file': False,
        'required_files': {},
        'patient_ids': False,
        'total_size': 0,
        'total_records': 0
    }

    # 1. 检查目录是否存在
    if not test_dir.exists():
        print(f"❌ 目录不存在: {test_dir}")
        return results
    else:
        print(f"✅ 目录存在: {test_dir}")
        results['exists'] = True

    # 2. 验证患者ID文件
    print(f"\n📋 验证患者ID文件...")
    id_file = test_dir / config['id_file']
    exists, ids_info = verify_patient_ids(test_dir, config['id_file'], config['id_vars'])

    if exists:
        print(f"✅ 患者ID文件: {config['id_file']}")
        for var, value in ids_info.items():
            if value is not None:
                print(f"  - {var}: {len(value)} 个患者")
            else:
                print(f"  - {var}: 未找到")
        results['patient_ids'] = True
    else:
        print(f"❌ 患者ID文件验证失败: {ids_info}")

    # 3. 验证必需文件
    print(f"\n📁 验证数据文件...")
    file_count = 0
    total_size = 0
    total_records = 0

    for file_pattern in config['required_files']:
        file_path = test_dir / file_pattern

        # 检查文件是否存在
        exists, size_mb = check_file_exists(file_path, file_pattern)

        if exists:
            file_count += 1
            total_size += size_mb

            # 验证parquet文件
            success, records, cols_or_error = verify_parquet_file(file_path, file_pattern)

            if success:
                print(f"  ✅ {file_pattern}: {records:,} 行, {cols_or_error} 列 ({size_mb:.1f} MB)")
                total_records += records
                results['required_files'][file_pattern] = {
                    'success': True,
                    'records': records,
                    'columns': cols_or_error,
                    'size_mb': size_mb
                }
            else:
                print(f"  ❌ {file_pattern}: 读取失败 - {cols_or_error}")
                results['required_files'][file_pattern] = {
                    'success': False,
                    'error': cols_or_error,
                    'size_mb': size_mb
                }
        else:
            print(f"  ❌ {file_pattern}: 文件不存在")
            results['required_files'][file_pattern] = {
                'success': False,
                'error': 'File not found'
            }

    results['total_size'] = total_size
    results['total_records'] = total_records

    # 4. 总结
    print(f"\n📊 {db_name.upper()} 验证总结:")
    print(f"  - 目录存在: ✅")
    print(f"  - 患者ID文件: {'✅' if results['patient_ids'] else '❌'}")
    print(f"  - 必需文件: {file_count}/{len(config['required_files'])} 个")
    print(f"  - 总记录数: {total_records:,}")
    print(f"  - 总大小: {total_size:.1f} MB")

    # 5. 特殊验证
    if db_name == 'hirid':
        # 检查HiRID特殊文件
        hirid_files = ['hirid_variable_reference.csv', 'ordinal_vars_ref.csv']
        for file_name in hirid_files:
            file_path = test_dir / file_name
            if file_path.exists():
                print(f"  ✅ {file_name}: 存在")
            else:
                print(f"  ❌ {file_name}: 缺失")

    return results

def main():
    print(f"🏥 pyricu 测试数据验证工具")
    print(f"{'='*60}")

    # 验证所有数据库
    all_results = {}
    for db_name, config in DATABASES.items():
        all_results[db_name] = verify_database(db_name, config)

    # 生成汇总报告
    print(f"\n{'='*60}")
    print(f"📋 验证汇总报告")
    print(f"{'='*60}")

    success_count = 0
    total_size = 0
    total_records = 0

    for db_name, results in all_results.items():
        if not results['exists']:
            status = "❌ 目录不存在"
        elif not results['patient_ids']:
            status = "❌ 患者ID缺失"
        elif not results['required_files']:
            status = "❌ 数据文件缺失"
        else:
            file_success = sum(1 for f in results['required_files'].values() if f.get('success', False))
            file_total = len(results['required_files'])

            if file_success == file_total:
                status = "✅ 完全成功"
                success_count += 1
            else:
                status = f"⚠️ 部分成功 ({file_success}/{file_total})"

        print(f"\n{DATABASES[db_name]['test_dir'].name:20} : {status}")

        if results['total_records'] > 0:
            total_records += results['total_records']
            total_size += results['total_size']
            print(f"{'':20}   {results['total_records']:,} 记录, {results['total_size']:.1f} MB")

    # 最终总结
    print(f"\n{'='*60}")
    print(f"🎯 最终结果")
    print(f"{'='*60}")
    print(f"数据库验证: {success_count}/{len(DATABASES)} 通过")
    print(f"总数据量: {total_records:,} 记录")
    print(f"总大小: {total_size:.1f} MB")

    if success_count == len(DATABASES):
        print(f"\n🎉 所有数据库验证通过！")
        print(f"\n📚 下一步:")
        print(f"  1. 运行 pyricu 加载测试:")
        print(f"     python test_main.py --database miiv --data-source test")
        print(f"  2. 运行 SOFA-2 评分测试:")
        print(f"     python -c \"import pyricu; print(pyricu.load_sofa2('miiv', patient_ids=[30057454]))\"")
    else:
        print(f"\n⚠️ 部分数据库验证失败，请检查上述错误信息")

if __name__ == "__main__":
    main()