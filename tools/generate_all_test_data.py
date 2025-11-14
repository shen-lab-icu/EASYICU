#!/usr/bin/env python3
"""统一生成所有数据库的测试数据"""

import subprocess
import sys
from pathlib import Path
import shutil
from datetime import datetime

# 工具目录路径
TOOLS_DIR = Path(__file__).resolve().parent
ROOT_DIR = TOOLS_DIR.parent

# 数据库配置
DATABASES = {
    'miiv': {
        'script': 'create_miiv_from_db.py',
        'test_dir': ROOT_DIR / 'test_data_miiv',
        'description': 'MIMIC-IV数据库'
    },
    'aumc': {
        'script': 'create_aumc_from_db.py',
        'test_dir': ROOT_DIR / 'test_data_aumc',
        'description': 'AUMC数据库'
    },
    'eicu': {
        'script': 'create_eicu_from_db.py',
        'test_dir': ROOT_DIR / 'test_data_eicu',
        'description': 'eICU数据库'
    },
    'hirid': {
        'script': 'create_hirid_from_db.py',
        'test_dir': ROOT_DIR / 'test_data_hirid',
        'description': 'HiRID数据库'
    }
}

def clean_existing_data(test_dir):
    """清理现有测试数据"""
    if test_dir.exists():
        print(f"  🗑️  清理现有目录: {test_dir}")
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)

def run_script(script_path, description):
    """运行生成脚本"""
    print(f"🚀 生成 {description} 测试数据...")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=1800  # 30分钟超时
        )

        if result.returncode == 0:
            print(f"  ✅ {description} 数据生成成功")
            # 输出关键信息
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['✅', '完成', '总计', '患者', '记录']):
                    print(f"    {line}")
        else:
            print(f"  ❌ {description} 数据生成失败")
            print(f"  错误信息: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"  ⏰ {description} 数据生成超时")
        return False
    except Exception as e:
        print(f"  ❌ {description} 数据生成出错: {e}")
        return False

    return True

def verify_generated_data(test_dir, description):
    """验证生成的数据"""
    print(f"🔍 验证 {description} 数据...")

    if not test_dir.exists():
        print(f"  ❌ 目录不存在: {test_dir}")
        return False

    # 统计parquet文件
    parquet_files = list(test_dir.rglob("*.parquet"))
    csv_files = list(test_dir.rglob("*.csv"))
    py_files = list(test_dir.rglob("*.py"))

    print(f"  📁 目录结构:")
    print(f"    - Parquet文件: {len(parquet_files)} 个")
    print(f"    - CSV文件: {len(csv_files)} 个")
    print(f"    - Python文件: {len(py_files)} 个")

    # 检查患者ID文件
    patient_ids_file = test_dir / "test_patient_ids.py"
    if patient_ids_file.exists():
        print(f"  ✅ 患者ID文件已生成: test_patient_ids.py")
    else:
        print(f"  ⚠️  患者ID文件缺失")

    # 计算总数据量
    total_rows = 0
    total_size = 0

    for parquet_file in parquet_files:
        try:
            # 使用pandas读取获取行数
            import pandas as pd
            df = pd.read_parquet(parquet_file)
            total_rows += len(df)
            total_size += parquet_file.stat().st_size
        except Exception as e:
            print(f"  ⚠️  无法读取 {parquet_file.name}: {e}")

    if total_size > 0:
        size_mb = total_size / (1024 * 1024)
        print(f"  📊 数据统计:")
        print(f"    - 总记录数: {total_rows:,}")
        print(f"    - 总大小: {size_mb:.1f} MB")

    return len(parquet_files) > 0

def generate_summary_report(results):
    """生成汇总报告"""
    print(f"\n{'='*60}")
    print(f"📋 测试数据生成汇总报告")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    success_count = sum(1 for r in results.values() if r['success'])
    total_count = len(results)

    print(f"\n📊 总体结果: {success_count}/{total_count} 数据库成功")

    print(f"\n📈 详细结果:")
    for db, result in results.items():
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"  {DATABASES[db]['description']:15} : {status}")
        if result['error']:
            print(f"    错误: {result['error']}")

    if success_count == total_count:
        print(f"\n🎉 所有数据库测试数据生成完成！")
        print(f"\n📂 生成的数据目录:")
        for db in results:
            test_dir = DATABASES[db]['test_dir']
            if test_dir.exists():
                parquet_count = len(list(test_dir.rglob("*.parquet")))
                print(f"  - {test_dir.name}: {parquet_count} 个parquet文件")
    else:
        print(f"\n⚠️  部分数据库生成失败，请检查错误信息")

    print(f"\n🔧 使用方法:")
    print(f"  # 导入患者ID")
    print(f"  from test_data_miiv.test_patient_ids import SELECTED_STAY_IDS")
    print(f"  # 加载测试数据")
    print(f"  python test_main.py --database miiv --data-source test")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='pyricu测试数据统一生成工具')
    parser.add_argument(
        'databases',
        nargs='*',
        help='要生成的数据库列表 (例如: miiv eicu)，默认: all'
    )
    parser.add_argument(
        '--auto-confirm',
        action='store_true',
        help='自动确认操作，不询问用户'
    )

    args = parser.parse_args()

    print(f"🏥 pyricu 测试数据统一生成工具")
    print(f"{'='*60}")

    # 确定要生成的数据库
    if not args.databases:
        selected_dbs = list(DATABASES.keys())  # 默认生成所有数据库
    else:
        if args.databases[0].lower() == 'all':
            selected_dbs = list(DATABASES.keys())
        else:
            selected_dbs = args.databases

    # 验证输入
    invalid_dbs = [db for db in selected_dbs if db not in DATABASES]
    if invalid_dbs:
        print(f"❌ 无效的数据库: {invalid_dbs}")
        print(f"可用的数据库: {', '.join(DATABASES.keys())}")
        return

    print(f"\n🎯 将为以下数据库生成测试数据: {', '.join(selected_dbs)}")

    # 显示数据库信息
    print(f"\n📋 数据库信息:")
    for db in selected_dbs:
        config = DATABASES[db]
        print(f"  {db}: {config['description']}")

    # 确认操作
    if not args.auto_confirm:
        print(f"\n⚠️  这将删除现有测试数据并重新生成")
        print(f"如果确认，请使用 --auto-confirm 参数")
        return

    # 执行生成
    results = {}

    for db in selected_dbs:
        config = DATABASES[db]
        print(f"\n{'-'*40}")

        results[db] = {'success': False, 'error': None}

        try:
            # 1. 清理现有数据
            clean_existing_data(config['test_dir'])

            # 2. 运行生成脚本
            script_path = TOOLS_DIR / config['script']
            if not script_path.exists():
                results[db]['error'] = f"脚本不存在: {script_path}"
                continue

            success = run_script(script_path, config['description'])
            if not success:
                results[db]['error'] = "脚本执行失败"
                continue

            # 3. 验证生成结果
            if verify_generated_data(config['test_dir'], config['description']):
                results[db]['success'] = True
            else:
                results[db]['error'] = "数据验证失败"

        except Exception as e:
            results[db]['error'] = str(e)
            print(f"  ❌ 处理 {config['description']} 时出错: {e}")

    # 生成汇总报告
    generate_summary_report(results)

if __name__ == "__main__":
    main()