#!/usr/bin/env python3
"""
高效提取 SOFA/SOFA2/susp_inf 数据并生成 Sepsis-3 事件

使用 DuckDB 加速的 pyricu 进行数据提取，显著提升性能。
整合了 Sepsis-3 事件检测，一次性完成数据提取和队列定义。

使用方法:
    # 提取单个数据库（默认 miiv）
    python extract_sofa_data.py --limit 5000
    
    # 提取多个数据库
    python extract_sofa_data.py --databases miiv,eicu,aumc --limit 5000
    
    # 提取全部患者
    python extract_sofa_data.py --limit -1
    
    # 指定输出目录（会自动创建子目录 miiv/, eicu/, aumc/）
    python extract_sofa_data.py --databases miiv,eicu,aumc --output sofa2_analysis/data
    
    # 跳过 sepsis 检测（仅提取数据）
    python extract_sofa_data.py --limit 5000 --no-sepsis
"""

import sys
import time
import argparse
from pathlib import Path
import os
import logging

import pandas as pd

# 设置路径
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "pyricu" / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# 禁用自动缓存清除以提升性能
os.environ['PYRICU_AUTO_CLEAR_CACHE'] = 'False'

# 设置 pyricu 日志级别为 WARNING，隐藏 INFO 日志
logging.getLogger('pyricu').setLevel(logging.WARNING)

from pyricu import load_concepts


# 数据库配置
DATABASE_CONFIG = {
    'miiv': {
        'data_path': '/home/1_publicData/icu_databases/mimiciv/3.1',
        'id_column': 'stay_id',
        'icustays_file': 'icustays.parquet',
        'total_patients': 94458,
    },
    'eicu': {
        'data_path': '/home/1_publicData/icu_databases/eicu/2.0.1',
        'id_column': 'patientunitstayid',
        'icustays_file': 'patient.parquet',
        'total_patients': 200859,
    },
    'aumc': {
        'data_path': '/home/1_publicData/icu_databases/aumc/1.0.2',
        'id_column': 'admissionid',
        'icustays_file': 'admissions.parquet',
        'total_patients': 23106,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="高效提取 SOFA/SOFA2/susp_inf 数据并生成 Sepsis-3 事件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=5000,
        help='患者数量限制，-1 表示全部 (默认: 5000)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='sofa2_analysis/data',
        help='输出目录 (默认: sofa2_analysis/data)'
    )
    parser.add_argument(
        '--databases', '-d',
        type=str,
        default='miiv',
        help='数据库名称，逗号分隔 (默认: miiv，可选: miiv,eicu,aumc)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='1h',
        help='时间间隔 (默认: 1h)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='并行工作进程数 (默认: 自动)'
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='禁用进度条'
    )
    parser.add_argument(
        '--no-sepsis',
        action='store_true',
        help='跳过 Sepsis-3 事件检测（仅提取原始数据）'
    )
    parser.add_argument(
        '--time-window',
        type=float,
        default=48.0,
        help='脓毒症检测时间窗口（小时），只考虑入院后这段时间内的脓毒症 (默认: 48)'
    )
    return parser.parse_args()


def get_patient_ids(database: str, limit: int = None) -> tuple:
    """获取患者 ID 列表
    
    Returns:
        tuple: (patient_ids, id_column, data_path)
    """
    config = DATABASE_CONFIG[database]
    data_path = Path(config['data_path'])
    id_column = config['id_column']
    icustays_file = data_path / config['icustays_file']
    
    if not icustays_file.exists():
        raise FileNotFoundError(f"找不到 ICU 入院文件: {icustays_file}")
    
    df = pd.read_parquet(icustays_file, columns=[id_column])
    ids = df[id_column].dropna().astype(int).unique().tolist()
    ids.sort()
    
    if limit and limit > 0:
        ids = ids[:limit]
    
    return ids, id_column, data_path


def standardize(table: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """确保 stay_id/charttime/value 列存在且命名一致"""
    if table is None or getattr(table, "empty", True):
        return pd.DataFrame(columns=["stay_id", "charttime", value_column])

    df = table.copy(deep=False)
    id_candidates = ["stay_id", "subject_id", "patientunitstayid", "admissionid", "patientid"]
    # 添加 AUMC 的时间列名 (measuredat, start) 和 eICU 的 offset 列名
    time_candidates = ["charttime", "measuredat", "start", "index_var", "time", "starttime", 
                       "diagnosisoffset", "infusionoffset", "labresultoffset", "observationoffset",
                       "nursingchartoffset", "respchartoffset", "intakeoutputoffset"]
    # eICU offset 列是分钟为单位，需要转换为小时
    eicu_offset_cols = ["diagnosisoffset", "infusionoffset", "labresultoffset", "observationoffset",
                        "nursingchartoffset", "respchartoffset", "intakeoutputoffset"]
    
    id_col = next((c for c in id_candidates if c in df.columns), df.columns[0])
    time_col = next((c for c in time_candidates if c in df.columns), None)
    value_col = value_column if value_column in df.columns else df.columns[-1]

    keep = [id_col]
    if time_col:
        keep.append(time_col)
    keep.append(value_col)

    df = df[keep].rename(columns={id_col: "stay_id", value_col: value_column})
    if time_col:
        df = df.rename(columns={time_col: "charttime"})
        # 如果是 eICU 的 offset 列（分钟），转换为小时
        if time_col in eicu_offset_cols:
            df["charttime"] = df["charttime"] / 60.0  # 分钟转小时
    else:
        df["charttime"] = pd.NA

    return df.dropna(subset=["stay_id"]).reset_index(drop=True)


def earliest_positive(events: pd.DataFrame, column: str) -> pd.Series:
    """返回每个 stay 的第一个阳性时间戳"""
    if events is None or column not in events.columns:
        return pd.Series(dtype="float64", name=f"{column}_onset")
    df = events[events[column].astype(bool)]
    if df.empty:
        return pd.Series(dtype="float64", name=f"{column}_onset")
    return df.groupby("stay_id")["charttime"].min().rename(f"{column}_onset")


def run_sepsis_detection(sofa: pd.DataFrame, sofa2: pd.DataFrame, susp_inf: pd.DataFrame, output_dir: Path, time_window_hours: float = 48.0):
    """运行 Sepsis-3 检测并保存结果
    
    Args:
        sofa: SOFA 评分数据
        sofa2: SOFA-2 评分数据
        susp_inf: 疑似感染数据
        output_dir: 输出目录
        time_window_hours: 时间窗口（小时），只考虑 ICU 入院后这个时间范围内的脓毒症
                          默认 48 小时，确保不同数据库的公平比较
    """
    from pyricu.sepsis import sep3 as sep3_detector
    from pyricu.sepsis_sofa2 import sep3_sofa2 as sep3_sofa2_detector
    
    print(f"\n🔬 运行 Sepsis-3 检测 (时间窗口: 入院后 {time_window_hours:.0f} 小时内)...")
    start_time = time.time()
    
    # 标准化数据
    sofa_df = standardize(sofa, "sofa")
    sofa2_df = standardize(sofa2, "sofa2")
    susp_df = standardize(susp_inf, "susp_inf")
    
    # 🔧 关键修改：只保留入院后 time_window_hours 小时内的数据
    # charttime 是相对于 ICU 入院的小时数，0 = 入院时刻
    print(f"   筛选时间窗口: charttime ∈ [0, {time_window_hours}] 小时")
    
    sofa_before = len(sofa_df)
    sofa2_before = len(sofa2_df)
    susp_before = len(susp_df)
    
    sofa_df = sofa_df[(sofa_df['charttime'] >= 0) & (sofa_df['charttime'] <= time_window_hours)].copy()
    sofa2_df = sofa2_df[(sofa2_df['charttime'] >= 0) & (sofa2_df['charttime'] <= time_window_hours)].copy()
    susp_df = susp_df[(susp_df['charttime'] >= 0) & (susp_df['charttime'] <= time_window_hours)].copy()
    
    print(f"   SOFA 数据: {sofa_before:,} → {len(sofa_df):,} 行 ({len(sofa_df)/sofa_before*100:.1f}%)")
    print(f"   SOFA2 数据: {sofa2_before:,} → {len(sofa2_df):,} 行 ({len(sofa2_df)/sofa2_before*100:.1f}%)")
    print(f"   susp_inf 数据: {susp_before:,} → {len(susp_df):,} 行 ({len(susp_df)/susp_before*100:.1f}%)")
    
    # 运行检测器
    sep3_events = sep3_detector(
        sofa=sofa_df,
        susp_inf=susp_df,
        id_cols=["stay_id"],
        index_col="charttime",
    )
    
    sep3_sofa2_events = sep3_sofa2_detector(
        sofa2=sofa2_df,
        susp_inf_df=susp_df,
        id_cols=["stay_id"],
        index_col="charttime",
    )
    
    elapsed = time.time() - start_time
    print(f"   检测耗时: {elapsed:.2f} 秒")
    
    # 保存 sep3 事件
    print("\n💾 保存 Sepsis-3 事件...")
    sep3_events.to_parquet(output_dir / "sep3_events.parquet", index=False)
    sep3_sofa2_events.to_parquet(output_dir / "sep3_sofa2_events.parquet", index=False)
    
    sep3_count = sep3_events['sep3'].sum() if 'sep3' in sep3_events.columns else 0
    sep3_sofa2_count = sep3_sofa2_events['sep3_sofa2'].sum() if 'sep3_sofa2' in sep3_sofa2_events.columns else 0
    
    print(f"   传统 SOFA sepsis 事件: {sep3_count:,}")
    print(f"   SOFA-2 sepsis 事件:    {sep3_sofa2_count:,}")
    
    # 生成队列比较
    print("\n📊 生成队列比较...")
    sofa_onset = earliest_positive(sep3_events, "sep3")
    sofa2_onset = earliest_positive(sep3_sofa2_events, "sep3_sofa2")
    
    comparison = pd.concat([sofa_onset, sofa2_onset], axis=1)
    comparison["status"] = comparison.apply(
        lambda row: (
            "both" if pd.notna(row.get("sep3_onset")) and pd.notna(row.get("sep3_sofa2_onset"))
            else "sofa_only" if pd.notna(row.get("sep3_onset"))
            else "sofa2_only" if pd.notna(row.get("sep3_sofa2_onset"))
            else "neither"
        ),
        axis=1,
    )
    comparison["onset_delta_h"] = comparison.apply(
        lambda row: row["sep3_sofa2_onset"] - row["sep3_onset"]
        if pd.notna(row.get("sep3_onset")) and pd.notna(row.get("sep3_sofa2_onset"))
        else pd.NA,
        axis=1,
    )
    comparison = comparison.reset_index()
    comparison.to_parquet(output_dir / "sepsis_cohort_comparison.parquet", index=False)
    
    # 输出统计
    print("\n" + "=" * 70)
    print("📊 队列比较统计")
    print("=" * 70)
    summary = comparison["status"].value_counts()
    for status, count in summary.items():
        print(f"   {status}: {count:,}")
    
    return {
        "sep3_events": sep3_events,
        "sep3_sofa2_events": sep3_sofa2_events,
        "comparison": comparison,
    }


def extract_single_database(
    database: str,
    output_dir: Path,
    limit: int,
    interval: str,
    workers: int,
    no_progress: bool,
    no_sepsis: bool,
    time_window_hours: float = 48.0,
) -> dict:
    """提取单个数据库的数据
    
    Args:
        time_window_hours: 脓毒症检测时间窗口（小时），只考虑入院后这段时间内的脓毒症
    """
    
    print(f"\n{'=' * 70}")
    print(f"🏥 处理数据库: {database.upper()}")
    print("=" * 70)
    
    # 获取患者 ID 和配置
    patient_ids, id_column, data_path = get_patient_ids(database, limit)
    config = DATABASE_CONFIG[database]
    
    print(f"\n📊 配置信息:")
    print(f"   数据库: {database}")
    print(f"   数据路径: {data_path}")
    print(f"   ID 列: {id_column}")
    print(f"   患者数量: {len(patient_ids):,}")
    print(f"   输出目录: {output_dir}")
    print(f"   时间间隔: {interval}")
    print(f"   Sepsis检测: {'禁用' if no_sepsis else '启用'}")
    
    # 确定并行配置
    if workers:
        actual_workers = workers
    elif len(patient_ids) < 2000:
        actual_workers = 1  # 小规模用单线程
    elif len(patient_ids) < 10000:
        actual_workers = 8
    else:
        actual_workers = 16
    
    backend = "thread" if actual_workers == 1 else "process"
    
    # 计算分块大小
    # 关键发现：当 chunk 数量 > workers 时，会导致卡死问题
    # 解决方案：确保 chunk 数量 ≤ workers，每个 worker 处理一个 chunk
    chunk_size = None
    if actual_workers > 1 and len(patient_ids) > 500:
        # 确保 chunk 数量 = workers，避免多轮调度导致的卡死
        chunk_size = max(1000, (len(patient_ids) + actual_workers - 1) // actual_workers)
    
    print(f"   并行工作进程: {actual_workers}")
    print(f"   并行后端: {backend}")
    print(f"   分块大小: {chunk_size if chunk_size else '禁用'}")
    
    # 准备加载参数
    loader_kwargs = {
        "database": database,
        "data_path": str(data_path),
        "interval": interval,
        "merge": False,
        "keep_components": True,
        "use_sofa2": True,
        "progress": not no_progress,
        "parallel_workers": actual_workers,
        "parallel_backend": backend,
        "concept_workers": 1,
    }
    
    if chunk_size:
        loader_kwargs["chunk_size"] = chunk_size
    
    # 开始提取
    print(f"\n🔄 开始提取数据...")
    # 加载基础概念
    concepts = ["sofa", "sofa2", "susp_inf"]
    
    start_time = time.time()
    
    results = load_concepts(
        concepts,
        patient_ids={id_column: patient_ids},
        **loader_kwargs,
    )
    
    extract_elapsed = time.time() - start_time
    
    # 提取结果
    def extract_frame(name: str) -> pd.DataFrame:
        if isinstance(results, dict):
            frame = results.get(name)
        else:
            frame = None
        
        if frame is None:
            return pd.DataFrame()
        if hasattr(frame, "dataframe"):
            return frame.dataframe()
        if hasattr(frame, "data"):
            return frame.data.copy()
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    
    sofa = extract_frame("sofa")
    sofa2 = extract_frame("sofa2")
    susp_inf = extract_frame("susp_inf")
    
    # 保存原始数据
    print(f"\n💾 保存原始数据...")
    sofa.to_parquet(output_dir / "sofa.parquet", index=False)
    sofa2.to_parquet(output_dir / "sofa2.parquet", index=False)
    susp_inf.to_parquet(output_dir / "susp_inf.parquet", index=False)
    
    # 输出提取统计
    print("\n" + "=" * 70)
    print(f"📊 [{database.upper()}] 数据提取结果")
    print("=" * 70)
    print(f"   SOFA:     {len(sofa):>10,} 行")
    print(f"   SOFA2:    {len(sofa2):>10,} 行")
    print(f"   susp_inf: {len(susp_inf):>10,} 行")
    print(f"   总计:     {len(sofa) + len(sofa2) + len(susp_inf):>10,} 行")
    print()
    print(f"⏱️  提取耗时: {extract_elapsed:.2f} 秒 ({extract_elapsed/60:.1f} 分钟)")
    print(f"📈 速度: {len(patient_ids)/extract_elapsed:.1f} 患者/秒")
    
    # 运行 Sepsis-3 检测
    sepsis_results = None
    if not no_sepsis:
        sepsis_results = run_sepsis_detection(sofa, sofa2, susp_inf, output_dir, time_window_hours)
    
    # 总耗时
    total_elapsed = time.time() - start_time
    
    print(f"\n✅ [{database.upper()}] 完成!")
    print(f"   总耗时: {total_elapsed:.2f} 秒 ({total_elapsed/60:.1f} 分钟)")
    print(f"\n📁 输出文件 ({output_dir}):")
    print("   - sofa.parquet")
    print("   - sofa2.parquet")
    print("   - susp_inf.parquet")
    if not no_sepsis:
        print("   - sep3_events.parquet")
        print("   - sep3_sofa2_events.parquet")
        print("   - sepsis_cohort_comparison.parquet")
    
    # 预估全库时间
    if limit and limit > 0:
        total_patients = config['total_patients']
        estimated_full = total_elapsed * (total_patients / len(patient_ids))
        print(f"\n📊 预估全库 ({total_patients:,} 患者) 处理时间: {estimated_full/60:.1f} 分钟")
    
    return {
        "database": database,
        "sofa": sofa,
        "sofa2": sofa2,
        "susp_inf": susp_inf,
        "sepsis_results": sepsis_results,
        "extract_elapsed": extract_elapsed,
        "total_elapsed": total_elapsed,
        "patient_count": len(patient_ids),
    }


def main():
    args = parse_args()
    
    # 解析数据库列表
    databases = [db.strip() for db in args.databases.split(',')]
    
    # 验证数据库名称
    for db in databases:
        if db not in DATABASE_CONFIG:
            print(f"❌ 未知数据库: {db}")
            print(f"   支持的数据库: {', '.join(DATABASE_CONFIG.keys())}")
            sys.exit(1)
    
    limit = None if args.limit < 0 else args.limit
    
    print("=" * 70)
    print("🚀 SOFA/SOFA2/susp_inf 数据提取 & Sepsis-3 事件检测")
    print("=" * 70)
    print(f"\n📋 待处理数据库: {', '.join(databases)}")
    
    all_results = {}
    global_start = time.time()
    
    for database in databases:
        # 每个数据库的输出子目录
        output_dir = PROJECT_ROOT / args.output / database
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = extract_single_database(
            database=database,
            output_dir=output_dir,
            limit=limit,
            interval=args.interval,
            workers=args.workers,
            no_progress=args.no_progress,
            no_sepsis=args.no_sepsis,
            time_window_hours=args.time_window,
        )
        all_results[database] = result
    
    # 总结
    global_elapsed = time.time() - global_start
    
    print("\n" + "=" * 70)
    print("🎉 全部完成!")
    print("=" * 70)
    print(f"\n📊 汇总统计:")
    for db, result in all_results.items():
        print(f"   [{db.upper()}] {result['patient_count']:,} 患者, "
              f"耗时 {result['total_elapsed']:.1f}s")
    print(f"\n⏱️  总耗时: {global_elapsed:.2f} 秒 ({global_elapsed/60:.1f} 分钟)")
    
    return all_results


if __name__ == "__main__":
    main()
