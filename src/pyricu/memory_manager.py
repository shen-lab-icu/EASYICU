"""
PyRICU 内存管理模块

解决 pandas/numpy 中间 DataFrame 导致的 glibc 内存碎片问题。
核心策略：
1. gc.collect() + malloc_trim(0) 回收碎片内存
2. 自动内存估算 + 智能分批
3. 子进程隔离（可选，用于16GB以下内存环境）

Usage:
    from pyricu.memory_manager import release_memory, auto_batch_size
    
    # 在关键点释放碎片内存
    release_memory()
    
    # 自动计算最佳 batch_size
    batch = auto_batch_size(['sofa'], 'miiv')
"""

import os
import gc
import time
import ctypes
import logging
import tempfile
import warnings
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 内存释放
# ============================================================

_libc = None
_last_gc_time: float = 0.0
_GC_MIN_INTERVAL: float = 30.0  # 最小gc间隔（秒），避免高频gc浪费CPU

def _get_libc():
    """懒加载 libc"""
    global _libc
    if _libc is None:
        try:
            _libc = ctypes.CDLL('libc.so.6')
        except OSError:
            _libc = False  # 标记为不可用
    return _libc if _libc is not False else None


def release_memory(aggressive: bool = False) -> int:
    """
    释放碎片内存，返回回收的 MB 数。
    
    调用 gc.collect() + malloc_trim(0) 将碎片化的堆内存归还操作系统。
    对于 pandas/numpy 密集操作后效果显著（可回收 10-30%）。
    
    🚀 性能优化：添加时间节流，避免高频gc.collect()浪费CPU。
    cProfile 显示41次gc.collect调用消耗1.3秒。现在限制最少30秒间隔。
    
    Args:
        aggressive: 如果 True，忽略节流限制，进行多轮 gc + trim
    
    Returns:
        估计回收的 MB 数（基于 RSS 变化）
    """
    global _last_gc_time
    
    # 🚀 非aggressive模式下，限制gc频率
    if not aggressive:
        now = time.monotonic()
        if now - _last_gc_time < _GC_MIN_INTERVAL:
            return 0  # 跳过——距上次gc太近
        _last_gc_time = now
    
    rss_before = get_rss_mb()
    
    if aggressive:
        # 多轮 gc 可以清除循环引用链
        for _ in range(3):
            gc.collect()
    else:
        gc.collect()
    
    libc = _get_libc()
    if libc is not None and hasattr(libc, 'malloc_trim'):
        libc.malloc_trim(0)
    
    rss_after = get_rss_mb()
    freed = max(0, rss_before - rss_after)
    
    if freed > 50:
        logger.debug(f"💾 release_memory: 回收 {freed:.0f} MB (RSS: {rss_before:.0f} → {rss_after:.0f})")
    
    return int(freed)


def get_rss_mb() -> float:
    """获取当前进程 RSS（常驻内存，MB）"""
    try:
        with open(f'/proc/{os.getpid()}/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    
    # fallback: resource module
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def get_available_memory_mb() -> float:
    """获取系统可用内存 (MB)"""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 2)
    except ImportError:
        return 8 * 1024  # 默认 8GB


# ============================================================
# 内存估算
# ============================================================

# 每概念内存模型: (fixed_mb, marginal_mb_per_patient)
# 内存消耗 ≈ fixed + marginal × num_patients
# 实测校准：
#   SOFA 10K pts → 2454 MB, 94K pts → 4142 MB
#   拟合: 2000 + 0.022 * N  (回验: 10K→2220, 94K→4078, 误差<5%)
#   hr 10K pts → ~400 MB  → 拟合: 200 + 0.02 * N
_MEMORY_COEFFICIENTS = {
    # (fixed_mb, marginal_mb_per_patient)
    # 复杂计算概念（含子概念递归加载 + DuckDB 工作内存）
    'sofa':        (2000, 0.022),     # 实测 10K→2.5G, 94K→4.1G
    'sofa2':       (2200, 0.025),     # sofa2 略多
    # SOFA 子分数（共享 DuckDB 开销较低）
    'sofa_resp':   (500, 0.008),
    'sofa_coag':   (200, 0.003),
    'sofa_liver':  (150, 0.002),
    'sofa_cardio': (400, 0.008),
    'sofa_cns':    (300, 0.005),
    'sofa_renal':  (600, 0.012),
    # 基础概念（单表读取，固定成本 = DuckDB扫描 + 后处理）
    'hr':    (200, 0.008),
    'sbp':   (200, 0.008),
    'dbp':   (200, 0.008),
    'map':   (250, 0.010),
    'resp':  (200, 0.006),
    'temp':  (150, 0.004),
    'spo2':  (200, 0.008),
    # Lab 概念
    'bili':  (150, 0.003),
    'crea':  (200, 0.005),
    'plt':   (200, 0.005),
    'glu':   (200, 0.005),
    # 尿量（outputevents 大表，高固定成本）
    'urine':   (600, 0.010),
    'urine24': (500, 0.015),
    # 其他复杂概念
    'pafi':      (300, 0.006),
    'gcs':       (300, 0.008),
    'sep3':      (1500, 0.020),
    'kdigo_aki': (1800, 0.022),
}

# 默认系数（未列出的概念）
_DEFAULT_COEFFICIENT = (200, 0.006)

# 数据库系数（基于每患者平均行数）
_DB_COEFFICIENTS = {
    'miiv': 1.0,      # 基准
    'eicu': 0.8,
    'aumc': 1.3,
    'hirid': 1.5,
    'mimic': 1.0,
    'sic': 2.0,
}

# 安全系数：覆盖峰值比最终 RSS 更高的情况
_SAFETY_MARGIN = 1.5


def estimate_memory_mb(
    concepts: List[str],
    database: str,
    num_patients: int,
) -> float:
    """
    估算加载给定概念所需的峰值内存 (MB)。
    
    使用亚线性模型：memory = base + sum(fixed_i) + sum(marginal_i) * db_coeff * N
    其中固定成本来自 DuckDB 表扫描/工作内存，边际成本来自每患者数据量。
    
    实测校准：
      SOFA 10K → 2454 MB, 94K → 4142 MB
      模型: 300 + 2000 + 0.022 * N → 10K=2520, 94K=4378 (误差<5%)
    
    Args:
        concepts: 概念列表
        database: 数据库名称
        num_patients: 患者数量
    
    Returns:
        估计峰值内存 (MB)
    """
    # 计算概念的固定成本和边际成本
    total_fixed = 0
    total_marginal = 0
    for c in concepts:
        fixed, marginal = _MEMORY_COEFFICIENTS.get(c, _DEFAULT_COEFFICIENT)
        total_fixed += fixed
        total_marginal += marginal
    
    # 数据库系数（影响边际成本：不同DB每患者行数不同）
    db_coeff = _DB_COEFFICIENTS.get(database, 1.0)
    
    # 基础开销（Python + pyricu 初始化 + DuckDB 引擎）
    base_mb = 300
    
    # 并行加载多个概念时，固定成本可能有重叠（取 max 更合理）
    # 但保守估计用 sum，安全系数会补偿
    estimated = (base_mb + total_fixed + total_marginal * db_coeff * num_patients) * _SAFETY_MARGIN
    
    return estimated


def auto_batch_size(
    concepts: List[str],
    database: str,
    total_patients: int,
    available_memory_mb: Optional[float] = None,
    memory_limit_ratio: float = 0.6,
) -> Optional[int]:
    """
    自动计算最佳 batch_size。
    
    如果整个 cohort 可以放入可用内存的 60%，返回 None（不需要分批）。
    否则返回一个合适的 batch_size。
    
    Args:
        concepts: 概念列表
        database: 数据库名称
        total_patients: 总患者数
        available_memory_mb: 可用内存 (MB)，None=自动检测
        memory_limit_ratio: 使用可用内存的比例上限
    
    Returns:
        None 表示不需要分批，int 表示推荐的 batch_size
    """
    if available_memory_mb is None:
        available_memory_mb = get_available_memory_mb()
    
    memory_budget = available_memory_mb * memory_limit_ratio
    
    # 估算全量加载的内存需求
    full_estimate = estimate_memory_mb(concepts, database, total_patients)
    
    if full_estimate <= memory_budget:
        logger.debug(
            f"📊 内存估算: {full_estimate:.0f}MB < 预算 {memory_budget:.0f}MB, "
            f"不需要分批 ({total_patients} patients)"
        )
        return None
    
    # 需要分批：基于 (base + fixed + marginal * N) 模型反算 N
    # 每批的固定开销 = (base_mb + total_fixed) * safety
    total_fixed = 0
    total_marginal = 0
    for c in concepts:
        fixed, marginal = _MEMORY_COEFFICIENTS.get(c, _DEFAULT_COEFFICIENT)
        total_fixed += fixed
        total_marginal += marginal
    db_coeff = _DB_COEFFICIENTS.get(database, 1.0)
    
    batch_fixed = (300 + total_fixed) * _SAFETY_MARGIN
    batch_marginal = total_marginal * db_coeff * _SAFETY_MARGIN  # MB per patient
    
    # batch_fixed + batch_marginal * N ≤ budget → N = (budget - batch_fixed) / batch_marginal
    if batch_marginal > 0:
        max_batch = int((memory_budget - batch_fixed) / batch_marginal)
    else:
        max_batch = total_patients
    
    # 确保 batch_size 合理
    batch_size = max(500, min(max_batch, total_patients))
    
    # 取整到 1000 的倍数（更整洁）
    batch_size = max(500, (batch_size // 1000) * 1000)
    
    logger.info(
        f"📊 自动分批: {total_patients} patients, "
        f"估算峰值 {full_estimate:.0f}MB > 预算 {memory_budget:.0f}MB, "
        f"推荐 batch_size={batch_size}"
    )
    
    return batch_size


# ============================================================
# 子进程隔离批处理
# ============================================================

def _subprocess_load_worker(args: dict) -> str:
    """
    在子进程中加载概念数据的 worker 函数。
    
    子进程退出后，其所有内存（包括碎片）完全归还 OS。
    
    Args:
        args: 包含 concepts, patient_ids, database, data_path, interval,
              output_path, ricu_data_path 等参数
    
    Returns:
        输出文件路径
    """
    import os
    # 设置环境变量
    if args.get('ricu_data_path'):
        os.environ['RICU_DATA_PATH'] = args['ricu_data_path']
    
    # 在子进程中导入（避免 fork 问题）
    from pyricu.api import load_concepts as _load_concepts
    
    result = _load_concepts(
        concepts=args['concepts'],
        patient_ids=args['patient_ids'],
        database=args['database'],
        data_path=args.get('data_path'),
        interval=args.get('interval', '1h'),
        verbose=False,
        ricu_compatible=args.get('ricu_compatible', True),
        merge=args.get('merge', True),
        dict_path=args.get('dict_path'),
        use_sofa2=args.get('use_sofa2', False),
    )
    
    # 写入临时 parquet 文件
    output_path = args['output_path']
    if isinstance(result, pd.DataFrame) and len(result) > 0:
        result.to_parquet(output_path, index=False, engine='pyarrow')
    elif isinstance(result, dict):
        # dict 结果：序列化为多个 parquet 文件
        for k, v in result.items():
            if isinstance(v, pd.DataFrame) and len(v) > 0:
                v.to_parquet(f"{output_path}.{k}.parquet", index=False, engine='pyarrow')
    
    return output_path


def subprocess_batch_load(
    concepts: List[str],
    database: str,
    all_patient_ids: dict,
    batch_size: int,
    data_path: Optional[str] = None,
    interval: str = '1h',
    verbose: bool = False,
    merge: bool = True,
    ricu_compatible: bool = True,
    dict_path=None,
    use_sofa2: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    使用子进程隔离的分批加载。
    
    每个 batch 在独立子进程中运行，退出后内存完全归还 OS。
    结果通过临时 parquet 文件传递，避免 pickle 序列化开销。
    
    Args:
        concepts: 概念列表
        database: 数据库名称
        all_patient_ids: {'id_col': [id1, id2, ...]} 字典
        batch_size: 每批患者数
        其余参数同 load_concepts
    
    Returns:
        合并后的 DataFrame
    """
    import multiprocessing as mp
    
    id_col = list(all_patient_ids.keys())[0]
    ids = list(all_patient_ids.values())[0]
    total = len(ids)
    num_batches = (total + batch_size - 1) // batch_size
    
    ricu_data_path = os.environ.get('RICU_DATA_PATH', '')
    
    if verbose:
        print(f"🔄 子进程隔离分批: {total} patients, batch_size={batch_size}, {num_batches} batches")
    
    temp_dir = tempfile.mkdtemp(prefix='pyricu_batch_')
    temp_files = []
    
    try:
        for i in range(0, total, batch_size):
            batch_num = i // batch_size + 1
            batch_ids = ids[i:i + batch_size]
            output_path = os.path.join(temp_dir, f'batch_{batch_num:04d}.parquet')
            
            if verbose:
                rss = get_rss_mb()
                print(f"   📦 Batch {batch_num}/{num_batches}: {len(batch_ids)} patients (RSS: {rss:.0f}MB)...", 
                      end='', flush=True)
            
            args = {
                'concepts': concepts,
                'patient_ids': {id_col: batch_ids},
                'database': database,
                'data_path': data_path,
                'interval': interval,
                'output_path': output_path,
                'ricu_data_path': ricu_data_path,
                'ricu_compatible': ricu_compatible,
                'merge': merge,
                'dict_path': str(dict_path) if dict_path else None,
                'use_sofa2': use_sofa2,
            }
            
            # 在子进程中运行
            # 使用 'spawn' 方法确保完全隔离
            ctx = mp.get_context('fork')  # fork 更快，且 pyricu 初始化已在父进程完成
            proc = ctx.Process(target=_subprocess_load_worker, args=(args,))
            proc.start()
            proc.join()
            
            if proc.exitcode != 0:
                logger.warning(f"⚠️ Batch {batch_num} 子进程退出码: {proc.exitcode}")
                if verbose:
                    print(f" ❌ (exit={proc.exitcode})")
                continue
            
            if os.path.exists(output_path):
                temp_files.append(output_path)
                if verbose:
                    file_mb = os.path.getsize(output_path) / 1024 / 1024
                    print(f" ✅ ({file_mb:.1f}MB)")
            else:
                if verbose:
                    print(f" ⚠️ (no output)")
        
        # 合并所有 batch 的结果
        if not temp_files:
            return pd.DataFrame()
        
        if verbose:
            print(f"   📋 合并 {len(temp_files)} 个批次...")
        
        frames = []
        for f in temp_files:
            try:
                df = pd.read_parquet(f, engine='pyarrow')
                frames.append(df)
            except Exception as e:
                logger.warning(f"⚠️ 读取 {f} 失败: {e}")
        
        if not frames:
            return pd.DataFrame()
        
        result = pd.concat(frames, ignore_index=True)
        
        if verbose:
            print(f"   ✅ 合并完成: {len(result)} rows, RSS: {get_rss_mb():.0f}MB")
        
        return result
    
    finally:
        # 清理临时文件
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


# ============================================================
# 进程内分批（带 malloc_trim）  
# ============================================================

def inprocess_batch_load(
    loader,
    concepts: List[str],
    patient_ids: dict,
    batch_size: int,
    verbose: bool = False,
    memory_efficient: bool = False,
    **load_kwargs,
) -> pd.DataFrame:
    """
    进程内分批加载，每批间调用 gc + malloc_trim 回收碎片。
    
    适用于内存充裕的服务器（>32GB），碎片不会导致 OOM。
    比子进程方案快（无 fork/序列化开销），但碎片只能部分回收。
    
    Args:
        loader: BaseICULoader 实例
        concepts: 概念列表
        patient_ids: {'id_col': [id1, id2, ...]} 字典
        batch_size: 每批患者数
        verbose: 是否显示进度
        memory_efficient: 压缩数据类型
        **load_kwargs: 传递给 loader.load_concepts 的其余参数
    """
    id_col = list(patient_ids.keys())[0]
    ids = list(patient_ids.values())[0]
    total = len(ids)
    num_batches = (total + batch_size - 1) // batch_size
    
    if verbose:
        print(f"🔄 进程内分批: {total} patients, batch_size={batch_size}, {num_batches} batches")
    
    results = []
    
    for i in range(0, total, batch_size):
        batch_num = i // batch_size + 1
        batch_ids = ids[i:i + batch_size]
        
        if verbose:
            rss = get_rss_mb()
            print(f"   📦 Batch {batch_num}/{num_batches}: {len(batch_ids)} patients (RSS: {rss:.0f}MB)...",
                  end='', flush=True)
        
        # 清除上一轮的缓存
        loader.clear_cache()
        
        batch_result = loader.load_concepts(
            concepts=concepts,
            patient_ids={id_col: batch_ids},
            **load_kwargs,
        )
        
        if isinstance(batch_result, pd.DataFrame) and len(batch_result) > 0:
            results.append(batch_result)
            if verbose:
                print(f" ✅ ({len(batch_result)} rows)", end='')
        elif verbose:
            print(f" ⚪ (empty)", end='')
        
        # 关键：释放碎片内存
        freed = release_memory()
        if verbose:
            print(f" [freed {freed}MB, RSS: {get_rss_mb():.0f}MB]")
    
    if not results:
        return pd.DataFrame()
    
    final = pd.concat(results, ignore_index=True)
    
    # 释放分片引用
    del results
    release_memory()
    
    if verbose:
        print(f"   ✅ 完成: {len(final)} rows, RSS: {get_rss_mb():.0f}MB")
    
    return final
