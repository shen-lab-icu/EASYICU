"""
EasyICU 内存管理模块

解决 pandas/numpy 中间 DataFrame 导致的 glibc 内存碎片问题。
核心策略：
1. gc.collect() + malloc_trim(0) 回收碎片内存
2. 自动内存估算 + 智能分批
3. 子进程隔离（可选，用于16GB以下内存环境）

Usage:
    from easyicu.runtime.memory_manager import release_memory, auto_batch_size
    
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
from pathlib import Path
from typing import Optional, List, Union, Dict

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# 内存释放
# ============================================================

_libc = None
_pa_pool = None  # 懒加载 pyarrow default_memory_pool
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


def _get_pyarrow_pool():
    """懒加载 pyarrow default_memory_pool。若 pyarrow 不可用或 pool 无
    release_unused，则缓存 False，后续直接跳过。"""
    global _pa_pool
    if _pa_pool is None:
        try:
            import pyarrow as pa
            pool = pa.default_memory_pool()
            if hasattr(pool, 'release_unused'):
                _pa_pool = pool
            else:
                _pa_pool = False  # 老版本 pyarrow 不支持
        except Exception:
            _pa_pool = False
    return _pa_pool if _pa_pool is not False else None


def release_memory(aggressive: bool = False) -> int:
    """
    释放碎片内存，返回回收的 MB 数。
    
    调用 gc.collect() + malloc_trim(0) 将碎片化的堆内存归还操作系统。
    另外调用 pyarrow.default_memory_pool().release_unused() 释放 Arrow
    缓冲区未使用部分（DuckDB→Arrow→pandas 路径会留下大量 Arrow buffer，
    EasyICU profile 显示 pool.max_memory 单次可达 1.2 GB）。
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
    
    # 释放 pyarrow 默认 pool 中未被引用的 Arrow buffer（DuckDB 路径累积）
    pa_pool = _get_pyarrow_pool()
    if pa_pool is not None:
        try:
            pa_pool.release_unused()
        except Exception:
            pass
    
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
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports ru_maxrss in KiB; macOS/BSD reports bytes.
        if os.name == 'posix' and hasattr(os, 'uname') and os.uname().sysname == 'Darwin':
            return rss / (1024.0 ** 2)
        return rss / 1024.0
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
#
# 🔧 2026-05-11 重新校准（基于 MIMIC-IV 实测，data on /Volumes/新加卷）：
#   分桶+DuckDB pushdown 让重概念的 peak 主要由"内部 hash join 工作集"决定，
#   而不是患者数。轻概念的 peak 由 output DataFrame 行数决定 (~0.16 KB/row)。
#
#   实测样本：
#     hr 单概念：    N=10k →  669 MB,  N=30k → 1034 MB  → (300, 0.020)
#     vitals 7个：   N=10k →  830 MB,  N=30k → 1552 MB  → 每概念约 (50, 0.0036)
#     chemistry 21个: N=10k →  599 MB                   → 每概念约 (15, 0.0019)
#     sofa1_full 7个: N=3k → 1602 MB, N=10k → 1670 MB   → 几乎不随 N 增长！
#     sep3 单概念：   N=3k → 2025 MB                    → 大固定开销
#
#   结论：重概念用大固定成本+几乎零边际；轻概念用小固定成本+小边际。
_MEMORY_COEFFICIENTS = {
    # (fixed_mb, marginal_mb_per_patient)
    # 重概念：DuckDB 内部 hash join + 递归子概念加载，peak 几乎不随 N 增长
    'sofa':        (1700, 0.005),     # 实测 N=3k→1602, N=10k→1670（基本平）
    'sofa2':       (1800, 0.006),     # 比 sofa 略重
    'sep3':        (2000, 0.005),     # 实测 N=3k→2025
    'sep3_sofa1':  (2000, 0.005),
    'sep3_sofa2':  (2100, 0.006),
    'kdigo_aki':   (1500, 0.008),     # 时间窗口计算
    'aki_stage':   (1500, 0.008),
    # SOFA 子分数（共享 DuckDB 开销较低）
    'sofa_resp':   (400, 0.003),
    'sofa_coag':   (200, 0.002),
    'sofa_liver':  (150, 0.002),
    'sofa_cardio': (400, 0.004),
    'sofa_cns':    (250, 0.003),
    'sofa_renal':  (500, 0.005),
    'sofa2_resp':  (400, 0.003),
    'sofa2_coag':  (200, 0.002),
    'sofa2_liver': (150, 0.002),
    'sofa2_cardio':(400, 0.004),
    'sofa2_cns':   (250, 0.003),
    'sofa2_renal': (500, 0.005),
    # 高密度 vitals（chartevents，~500 rows/patient）
    'hr':    (300, 0.020),     # 实测 N=10k→669, N=30k→1034
    'sbp':   (250, 0.018),
    'dbp':   (250, 0.018),
    'map':   (250, 0.020),
    'resp':  (200, 0.015),
    'temp':  (200, 0.010),
    'spo2':  (250, 0.018),
    'etco2': (250, 0.015),
    # Lab 概念（labevents，~50-150 rows/patient，稀疏）
    'bili':  (100, 0.003),
    'crea':  (150, 0.005),
    'plt':   (150, 0.005),
    'glu':   (150, 0.005),
    'lact':  (200, 0.006),
    'k':     (100, 0.004),
    'na':    (100, 0.004),
    # 尿量
    'urine':   (400, 0.008),
    'urine24': (400, 0.010),
    # 其他
    'pafi':      (250, 0.005),
    'safi':      (250, 0.005),
    'gcs':       (250, 0.006),
    'mech_vent': (300, 0.005),
    'vent_ind':  (250, 0.004),
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

# 基础开销（Python + easyicu 初始化 + DuckDB 引擎），单位 MB
_BASE_MB = 300

# 工作集重叠系数（2026-06-11 重新校准）。
#
# 关键事实：`_MEMORY_COEFFICIENTS` 里的 (fixed, marginal) 是**单概念独立加载**
# 的峰值，而不是成组加载的增量。成组加载时各概念共享同一批源表扫描 / DuckDB
# hash 工作集，峰值由**最重的那个概念主导**，其余概念基本是叠加在已驻留工作集
# 上的少量增量——尤其是 score 组（如 sofa2 + 6 个子分数）：父概念 sofa2 递归
# 加载时已经把 6 个子分数全拉进来了，再把子分数列进概念列表只是重复计数。
#
# 因此组合成本不是 sum，而是 max + share*(sum-max)：
#   share=0 → 纯 max（完全重叠，适合"父+自身子分数"这类全冗余组）
#   share=1 → 纯 sum（完全独立，旧行为）
# 0.25 是在实测点上拟合出的折中：score 组接近 max（实测 peak 几乎不随概念数、
# 不随 N 增长），多个**独立**重概念（如 20 个 vitals）仍随概念数增长以保留真实
# OOM 保护。详见 `.tmp/mem_calib_verify.py` 对所有实测点的回归。
_WORKINGSET_OVERLAP_SHARE = 0.25


def _combine_overlap(values: List[float]) -> float:
    """把一组单概念成本组合成成组加载的等效成本。

    max(values) + share * (sum(values) - max(values))。
    见 `_WORKINGSET_OVERLAP_SHARE` 的说明：成组加载共享源表工作集，
    峰值由最重概念主导，其余按 share 折扣叠加。
    """
    if not values:
        return 0.0
    mx = max(values)
    rest = sum(values) - mx
    return mx + _WORKINGSET_OVERLAP_SHARE * rest


def _combined_coefficients(concepts: List[str]) -> tuple:
    """返回成组加载的 (combined_fixed_mb, combined_marginal_mb_per_patient)。

    `estimate_memory_mb` 与 `auto_batch_size` 共用此函数，保证两者的内存
    模型完全一致（反算 batch_size 时必须用同一组合规则）。
    """
    fixed_list = []
    marginal_list = []
    for c in concepts:
        fixed, marginal = _MEMORY_COEFFICIENTS.get(c, _DEFAULT_COEFFICIENT)
        fixed_list.append(fixed)
        marginal_list.append(marginal)
    return _combine_overlap(fixed_list), _combine_overlap(marginal_list)


def estimate_memory_mb(
    concepts: List[str],
    database: str,
    num_patients: int,
) -> float:
    """
    估算加载给定概念所需的峰值内存 (MB)。

    模型：memory = (base + combined_fixed + combined_marginal * db_coeff * N) * safety

    其中 combined_fixed / combined_marginal 由 `_combined_coefficients` 用
    "max + share*rest" 的重叠模型从单概念系数组合而来（见
    `_WORKINGSET_OVERLAP_SHARE`）。这反映实测的"成组加载共享工作集、峰值由
    最重概念主导"行为，而不是把各单概念峰值简单相加。

    实测校准（miiv，单次 in-process 流式加载，EASYICU_FORCE_INPROCESS_BATCH=1）：
      score 组 [sofa2 + 6 子分数] 的真实 peak ~2-2.85 GB，**几乎不随 N 增长**
      （10k/20k/40k/94k 基本持平，因为加载按源表流式而非一次性持有全部行）。
      旧的 sum 模型在 94k 估到 ~9.5 GB（3-5× 高估），会在任何可用内存 < 6 GB
      的机器上误触发低内存分批；新模型估到 ~5.4 GB，安全留有 1.5× 余量但不再
      在 6 GB 边缘误触发。

    Args:
        concepts: 概念列表
        database: 数据库名称
        num_patients: 患者数量

    Returns:
        估计峰值内存 (MB)
    """
    combined_fixed, combined_marginal = _combined_coefficients(concepts)

    # 数据库系数（影响边际成本：不同DB每患者行数不同）
    db_coeff = _DB_COEFFICIENTS.get(database, 1.0)

    estimated = (
        _BASE_MB
        + combined_fixed
        + combined_marginal * db_coeff * num_patients
    ) * _SAFETY_MARGIN

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
    
    # 需要分批：基于 (base + combined_fixed + combined_marginal * N) 模型反算 N。
    # 必须与 estimate_memory_mb 用同一组合规则（_combined_coefficients），否则
    # 反算出的 batch_size 与触发分批的估算自相矛盾。
    # 每批的固定开销 = (base_mb + combined_fixed) * safety，每批工作集相同。
    combined_fixed, combined_marginal = _combined_coefficients(concepts)
    db_coeff = _DB_COEFFICIENTS.get(database, 1.0)

    batch_fixed = (_BASE_MB + combined_fixed) * _SAFETY_MARGIN
    batch_marginal = combined_marginal * db_coeff * _SAFETY_MARGIN  # MB per patient
    
    # batch_fixed + batch_marginal * N ≤ budget → N = (budget - batch_fixed) / batch_marginal
    if batch_marginal > 0:
        max_batch = int((memory_budget - batch_fixed) / batch_marginal)
    else:
        max_batch = total_patients
    
    # 🔧 2026-05-11: 保底从 500 提升到 10000。
    # 实测（hr/vitals/sofa1_full 在 N=3k/10k/30k）显示，分桶+pushdown 让 DuckDB
    # 工作集（pool_max ~20-60 MB）几乎不随患者数变化，峰值由 output DataFrame 决定。
    # batch_size=500 没有内存意义，反而导致 ~190 次 subprocess fork 的巨大开销。
    # 即使在低内存系统上，10000 一批的单次峰值（实测 vitals 7-concept @ 10k = 830 MB）
    # 也远低于 12GB 系统的可用预算。
    MIN_BATCH = 10000
    batch_size = max(MIN_BATCH, min(max_batch, total_patients))
    
    # 取整到 1000 的倍数（更整洁）
    batch_size = max(MIN_BATCH, (batch_size // 1000) * 1000)
    
    # 极端情况：若全量已经小于保底批，则不分批
    if batch_size >= total_patients:
        logger.debug(
            f"📊 内存估算: batch_size ({batch_size}) >= total ({total_patients}), 不分批"
        )
        return None
    
    logger.info(
        f"📊 自动分批: {total_patients} patients, "
        f"估算峰值 {full_estimate:.0f}MB > 预算 {memory_budget:.0f}MB, "
        f"推荐 batch_size={batch_size}"
    )
    
    return batch_size


# ============================================================
# 子进程隔离批处理
# ============================================================

def _fork_and_run(target, args) -> int:
    """
    使用 os.fork() 在子进程中运行 target(args)。

    绕过 multiprocessing.Process 的 daemon 限制——daemon 进程不允许
    创建 mp.Process 子进程，但 os.fork() 不受此限制。
    子进程退出后 OS 完整回收所有内存（包括 pymalloc arena 碎片）。

    Args:
        target: 可调用对象，签名 target(args)
        args: 传给 target 的参数

    Returns:
        子进程退出码（0 表示成功）
    """
    pid = os.fork()
    if pid == 0:
        # ---- 子进程 ----
        try:
            target(args)
            os._exit(0)
        except SystemExit as e:
            os._exit(e.code if isinstance(e.code, int) else 1)
        except Exception:
            import traceback
            traceback.print_exc()
            os._exit(1)
    else:
        # ---- 父进程 ----
        # 2026-05-20 fix (Bug E6): the previous version called
        #   os.waitpid(pid, 0)
        # which blocks forever. If a child process hangs (deadlock in
        # DuckDB, stuck mutex, OOM-but-not-killed-yet, …) the parent
        # waits forever — observed on real benchmarks. Poll with
        # WNOHANG, and force-kill after a generous timeout so the
        # convoy can move on. Same envelope as _popen_and_run (1 h).
        import signal
        timeout_s = float(os.environ.get('EASYICU_BATCH_TIMEOUT_SEC', '3600'))
        start = time.time()
        while True:
            try:
                done_pid, status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                return -1
            if done_pid != 0:
                break
            if time.time() - start > timeout_s:
                logger.warning(
                    f"⚠️ child pid={pid} exceeded {timeout_s:.0f}s, terminating"
                )
                try:
                    os.kill(pid, signal.SIGTERM)
                    # give it a moment to flush
                    for _ in range(50):
                        done_pid, status = os.waitpid(pid, os.WNOHANG)
                        if done_pid != 0:
                            return -1
                        time.sleep(0.1)
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                except OSError:
                    pass
                return -1
            time.sleep(0.2)
        if hasattr(os, 'waitstatus_to_exitcode'):
            return os.waitstatus_to_exitcode(status)
        # Python < 3.9 fallback
        if os.WIFEXITED(status):
            return os.WEXITSTATUS(status)
        return -1


def _popen_and_run(args: dict, temp_dir: str, batch_num: int) -> int:
    """
    使用 subprocess.Popen 在完全独立的进程中运行 _subprocess_load_worker。

    解决 Windows daemon 进程无法创建 multiprocessing.Process 子进程的问题：
    - multiprocessing.Process.start() 在 daemon 中抛出 AssertionError
    - os.fork() 在 Windows 上不存在
    - subprocess.Popen (CreateProcess) 不受 daemon 限制

    子进程退出后 OS 完整回收所有内存（包括 pymalloc arena 碎片），
    提供与 Linux os.fork() 等价的内存隔离效果。

    Args:
        args: 传给 _subprocess_load_worker 的参数字典
        temp_dir: 临时目录路径
        batch_num: 批次编号

    Returns:
        子进程退出码（0 表示成功）
    """
    import subprocess
    import sys
    import json

    # 序列化 args 到 JSON 文件（patient_ids 可能含 numpy int64，需转换）
    args_file = os.path.join(temp_dir, f'_args_{batch_num:04d}.json')
    _json_safe = dict(args)
    if 'patient_ids' in _json_safe and _json_safe['patient_ids']:
        _json_safe['patient_ids'] = {
            k: [int(x) for x in v]
            for k, v in _json_safe['patient_ids'].items()
        }
    with open(args_file, 'w') as f:
        json.dump(_json_safe, f)

    # 确保子进程能 import easyicu（处理非 pip 安装的开发模式）
    env = os.environ.copy()
    _easyicu_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _existing_pp = env.get('PYTHONPATH', '')
    if _easyicu_parent not in _existing_pp:
        env['PYTHONPATH'] = _easyicu_parent + os.pathsep + _existing_pp

    _code = (
        "import sys, json; "
        "from easyicu.runtime.memory_manager import _subprocess_load_worker; "
        "f = open(sys.argv[1], encoding='utf-8'); args = json.load(f); f.close(); "
        "_subprocess_load_worker(args)"
    )

    try:
        result = subprocess.run(
            [sys.executable, '-c', _code, args_file],
            env=env,
            timeout=3600,  # 1 hour timeout per batch
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        logger.warning(f"⚠️ Batch {batch_num} Popen 超时 (60min)")
        return -1
    except Exception as e:
        logger.warning(f"⚠️ Batch {batch_num} Popen 失败: {e}")
        return -1
    finally:
        try:
            os.unlink(args_file)
        except Exception:
            pass


def _subprocess_load_worker(args: dict) -> str:
    """
    在子进程中加载概念数据的 worker 函数。
    
    子进程退出后，其所有内存（包括碎片）完全归还 OS。
    
    Args:
        args: 包含 concepts, patient_ids, database, data_path, interval,
              output_path, easyicu_data_path 等参数
    
    Returns:
        输出文件路径
    """
    import os
    # 设置环境变量
    if args.get('easyicu_data_path'):
        os.environ['EASYICU_DATA_PATH'] = args['easyicu_data_path']
    
    # 在子进程中导入（避免 fork 问题）
    from easyicu.api import load_concepts as _load_concepts
    
    load_kwargs = dict(args.get('load_kwargs') or {})

    result = _load_concepts(
        concepts=args['concepts'],
        patient_ids=args['patient_ids'],
        database=args['database'],
        data_path=args.get('data_path'),
        verbose=False,
        r_compatible=args.get('r_compatible', True),
        merge=args.get('merge', True),
        dict_path=args.get('dict_path'),
        use_sofa2=args.get('use_sofa2', False),
        **load_kwargs,
    )
    
    # 写入临时 parquet 文件
    output_prefix = args['output_prefix']
    if isinstance(result, pd.DataFrame) and len(result) > 0:
        result.to_parquet(f"{output_prefix}.parquet", index=False, engine='pyarrow')
    elif isinstance(result, dict):
        # dict 结果：序列化为多个 parquet 文件
        # 处理 ICUTable/TsTbl/WinTbl 等具有 .data 属性的表对象
        for k, v in result.items():
            df = v
            if hasattr(v, 'data') and isinstance(v.data, pd.DataFrame):
                df = v.data
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                df.to_parquet(f"{output_prefix}.{k}.parquet", index=False, engine='pyarrow')
    
    return output_prefix


def _estimate_result_size_mb(result: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> float:
    """Estimate in-memory size of a batch result in MB."""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            return 0.0
        return float(result.memory_usage(deep=True).sum()) / 1024.0 / 1024.0

    if isinstance(result, dict):
        total = 0.0
        for value in result.values():
            df = value
            if hasattr(value, 'data') and isinstance(value.data, pd.DataFrame):
                df = value.data
            if isinstance(df, pd.DataFrame) and not df.empty:
                total += float(df.memory_usage(deep=True).sum()) / 1024.0 / 1024.0
        return total

    return 0.0


def _write_batch_result_to_parquet(
    result: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    output_prefix: str,
) -> bool:
    """Persist a batch result to parquet shards.

    Returns True if at least one parquet file was written.
    """
    wrote = False
    if isinstance(result, pd.DataFrame):
        if not result.empty:
            result.to_parquet(f"{output_prefix}.parquet", index=False, engine='pyarrow')
            wrote = True
        return wrote

    if isinstance(result, dict):
        for name, frame in result.items():
            df = frame
            if hasattr(frame, 'data') and isinstance(frame.data, pd.DataFrame):
                df = frame.data
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_parquet(f"{output_prefix}.{name}.parquet", index=False, engine='pyarrow')
                wrote = True
    return wrote


def _should_spill_inprocess_batches(
    *,
    memory_efficient: bool,
    num_batches: int,
    estimated_total_mb: float,
    buffered_mb: float,
) -> bool:
    """Decide whether in-process batching should spill intermediate results to disk."""
    if num_batches <= 1:
        return False

    if memory_efficient:
        return True

    if estimated_total_mb <= 0 and buffered_mb <= 0:
        return False

    available_mb = get_available_memory_mb()
    spill_threshold_mb = min(max(available_mb * 0.25, 1024.0), 4096.0)
    return estimated_total_mb >= spill_threshold_mb or buffered_mb >= spill_threshold_mb


def _merge_buffered_batches(
    buffered_batches: List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Merge in-memory buffered batch results without writing temporary parquet files."""
    if not buffered_batches:
        return pd.DataFrame()

    first = next((item for item in buffered_batches if item is not None), None)
    if first is None:
        return pd.DataFrame()

    if isinstance(first, dict):
        grouped: Dict[str, List[pd.DataFrame]] = {}
        for batch in buffered_batches:
            if not isinstance(batch, dict):
                continue
            for name, frame in batch.items():
                # 处理 ICUTable/TsTbl/WinTbl 等具有 .data 属性的表对象
                df = frame
                if hasattr(frame, 'data') and isinstance(frame.data, pd.DataFrame):
                    df = frame.data
                if isinstance(df, pd.DataFrame) and not df.empty:
                    grouped.setdefault(name, []).append(df)
        return {
            name: pd.concat(frames, ignore_index=True, sort=False, copy=False)
            for name, frames in grouped.items()
            if frames
        }

    frames = [frame for frame in buffered_batches if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False, copy=False)


def _merge_parquet_batches(temp_dir: str) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Merge parquet shards produced by batch loading."""
    from collections import defaultdict
    import re

    temp_path = Path(temp_dir)
    flat_files = sorted(
        f for f in temp_path.glob('batch_*.parquet')
        if re.fullmatch(r'batch_\d{4}\.parquet', f.name)
    )
    dict_files = sorted(temp_path.glob('batch_*.*.parquet'))

    if flat_files:
        frames = []
        for f in flat_files:
            try:
                frames.append(pd.read_parquet(f, engine='pyarrow'))
            except Exception as e:
                logger.warning(f"⚠️ 读取 {f} 失败: {e}")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False, copy=False)

    grouped: Dict[str, List[Path]] = defaultdict(list)
    for f in dict_files:
        try:
            parts = f.name.split('.')
            if len(parts) < 3:
                continue
            concept = '.'.join(parts[1:-1])
            grouped[concept].append(f)
        except Exception as e:
            logger.warning(f"⚠️ 读取 {f} 失败: {e}")

    merged: Dict[str, pd.DataFrame] = {}
    for concept, files in grouped.items():
        frames = []
        for f in files:
            try:
                frames.append(pd.read_parquet(f, engine='pyarrow'))
            except Exception as e:
                logger.warning(f"⚠️ 读取 {f} 失败: {e}")
        if frames:
            merged[concept] = pd.concat(frames, ignore_index=True, sort=False, copy=False)
            del frames
            release_memory()

    return merged


def subprocess_batch_load(
    concepts: List[str],
    database: str,
    all_patient_ids: dict,
    batch_size: int,
    data_path: Optional[str] = None,
    verbose: bool = False,
    merge: bool = True,
    r_compatible: bool = True,
    dict_path=None,
    use_sofa2: bool = False,
    **kwargs,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
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
    
    easyicu_data_path = os.environ.get('EASYICU_DATA_PATH', '')

    # 检测是否在 daemon 进程中
    # daemon 进程不允许 mp.Process.start()，但 os.fork() 不受此限制
    _in_daemon = False
    try:
        _in_daemon = mp.current_process().daemon
    except Exception:
        pass
    _use_raw_fork = _in_daemon and hasattr(os, 'fork')
    # Windows daemon: 无 os.fork 也无 mp.Process → 用 subprocess.Popen
    _use_popen = _in_daemon and not hasattr(os, 'fork')
    
    if verbose:
        mode = "os.fork()" if _use_raw_fork else ("Popen" if _use_popen else "mp.Process")
        print(f"🔄 子进程隔离分批: {total} patients, batch_size={batch_size}, "
              f"{num_batches} batches [{mode}]")
    
    temp_dir = tempfile.mkdtemp(prefix='easyicu_batch_')
    try:
        for i in range(0, total, batch_size):
            batch_num = i // batch_size + 1
            batch_ids = ids[i:i + batch_size]
            output_prefix = os.path.join(temp_dir, f'batch_{batch_num:04d}')
            
            if verbose:
                rss = get_rss_mb()
                print(f"   📦 Batch {batch_num}/{num_batches}: {len(batch_ids)} patients (RSS: {rss:.0f}MB)...", 
                      end='', flush=True)
            
            args = {
                'concepts': concepts,
                'patient_ids': {id_col: batch_ids},
                'database': database,
                'data_path': data_path,
                'output_prefix': output_prefix,
                'easyicu_data_path': easyicu_data_path,
                'r_compatible': r_compatible,
                'merge': merge,
                'dict_path': str(dict_path) if dict_path else None,
                'use_sofa2': use_sofa2,
                'load_kwargs': kwargs,
            }
            
            # 在子进程中运行
            if _use_raw_fork:
                # daemon 进程 (Linux/macOS): 用 os.fork() 绕过 mp.Process 的 daemon 限制
                exitcode = _fork_and_run(_subprocess_load_worker, args)
            elif _use_popen:
                # daemon 进程 (Windows): 用 subprocess.Popen 绕过 daemon 限制
                # Popen 创建完全独立的进程（CreateProcess），不受 daemon 约束
                exitcode = _popen_and_run(args, temp_dir, batch_num)
            else:
                # 非 daemon: 用 mp.Process（更安全的管理方式）
                # Linux/macOS 使用 fork（快速），Windows 用 spawn
                _method = 'fork' if hasattr(os, 'fork') else 'spawn'
                ctx = mp.get_context(_method)
                proc = ctx.Process(target=_subprocess_load_worker, args=(args,))
                proc.start()
                # 2026-05-20 fix (Bug E6): bounded join — see _fork_and_run.
                _timeout = float(os.environ.get('EASYICU_BATCH_TIMEOUT_SEC', '3600'))
                proc.join(timeout=_timeout)
                if proc.is_alive():
                    logger.warning(
                        f"⚠️ Batch {batch_num} mp.Process hang past "
                        f"{_timeout:.0f}s, terminating"
                    )
                    proc.terminate()
                    proc.join(timeout=10)
                    if proc.is_alive():
                        proc.kill()
                        proc.join(timeout=5)
                    exitcode = -1
                else:
                    exitcode = proc.exitcode
            
            if exitcode != 0:
                logger.warning(f"⚠️ Batch {batch_num} 子进程退出码: {exitcode}")
                if verbose:
                    print(f" ❌ (exit={exitcode})")
                continue
            
            output_files = [f for f in Path(temp_dir).glob(f"batch_{batch_num:04d}*.parquet")]
            if output_files:
                if verbose:
                    file_mb = sum(os.path.getsize(f) for f in output_files) / 1024 / 1024
                    print(f" ✅ ({file_mb:.1f}MB)")
            else:
                if verbose:
                    print(" ⚠️ (no output)")
        
        # 合并所有 batch 的结果
        produced_files = list(Path(temp_dir).glob('batch_*.parquet')) + list(Path(temp_dir).glob('batch_*.*.parquet'))
        if not produced_files:
            return pd.DataFrame()
        
        if verbose:
            print("   📋 合并批次结果...")

        result = _merge_parquet_batches(temp_dir)
        
        if verbose:
            if isinstance(result, pd.DataFrame):
                print(f"   ✅ 合并完成: {len(result)} rows, RSS: {get_rss_mb():.0f}MB")
            else:
                total_rows = sum(len(df) for df in result.values())
                print(f"   ✅ 合并完成: {len(result)} concepts / {total_rows} rows, RSS: {get_rss_mb():.0f}MB")
        
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
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
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
    
    buffered_batches: List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = []
    buffered_mb = 0.0
    estimated_total_mb = 0.0
    representative_batch_mb = 0.0
    spill_dir: Optional[str] = None
    spill_batches = 0
    
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
        
        if spill_dir is None:
            if representative_batch_mb == 0.0:
                representative_batch_mb = _estimate_result_size_mb(batch_result)
                if representative_batch_mb > 0:
                    estimated_total_mb = representative_batch_mb * num_batches
            batch_result_mb = representative_batch_mb
        else:
            batch_result_mb = 0.0

        if isinstance(batch_result, pd.DataFrame) and len(batch_result) > 0:
            if verbose:
                print(f" ✅ ({len(batch_result)} rows)", end='')
        elif isinstance(batch_result, dict):
            if verbose:
                non_empty = 0
                for _v in batch_result.values():
                    _df = _v.data if hasattr(_v, 'data') and isinstance(_v.data, pd.DataFrame) else _v
                    if isinstance(_df, pd.DataFrame) and len(_df) > 0:
                        non_empty += len(_df)
                print(f" ✅ ({non_empty} rows / {len(batch_result)} concepts)", end='')
        elif verbose:
            print(" ⚪ (empty)", end='')

        if spill_dir is None and _should_spill_inprocess_batches(
            memory_efficient=memory_efficient,
            num_batches=num_batches,
            estimated_total_mb=estimated_total_mb,
            buffered_mb=buffered_mb + batch_result_mb,
        ):
            spill_dir = tempfile.mkdtemp(prefix='easyicu_inprocess_')
            if verbose:
                print(f" 💽 spill→disk[{spill_dir}]", end='')
            for buffered in buffered_batches:
                spill_batches += 1
                _write_batch_result_to_parquet(buffered, os.path.join(spill_dir, f'batch_{spill_batches:04d}'))
            buffered_batches.clear()
            buffered_mb = 0.0
            release_memory(aggressive=True)

        if spill_dir is not None:
            spill_batches += 1
            _write_batch_result_to_parquet(batch_result, os.path.join(spill_dir, f'batch_{spill_batches:04d}'))
            del batch_result
        elif (
            isinstance(batch_result, pd.DataFrame) and not batch_result.empty
        ) or isinstance(batch_result, dict):
            buffered_batches.append(batch_result)
            buffered_mb += batch_result_mb
        
        # 关键：释放碎片内存
        freed = release_memory()
        if verbose:
            print(f" [freed {freed}MB, RSS: {get_rss_mb():.0f}MB]")
    
    if not buffered_batches and spill_dir is None:
        return pd.DataFrame()

    if spill_dir is not None:
        try:
            final = _merge_parquet_batches(spill_dir)
        finally:
            import shutil
            shutil.rmtree(spill_dir, ignore_errors=True)
        release_memory(aggressive=True)
        if verbose:
            if isinstance(final, pd.DataFrame):
                print(f"   ✅ 完成(disk): {len(final)} rows, RSS: {get_rss_mb():.0f}MB")
            else:
                total_rows = sum(len(df) for df in final.values())
                print(f"   ✅ 完成(disk): {len(final)} concepts / {total_rows} rows, RSS: {get_rss_mb():.0f}MB")
        return final

    final = _merge_buffered_batches(buffered_batches)
    del buffered_batches
    release_memory(aggressive=True)

    if verbose:
        if isinstance(final, pd.DataFrame):
            print(f"   ✅ 完成: {len(final)} rows, RSS: {get_rss_mb():.0f}MB")
        else:
            total_rows = sum(len(df) for df in final.values())
            print(f"   ✅ 完成: {len(final)} concepts / {total_rows} rows, RSS: {get_rss_mb():.0f}MB")

    return final


def inprocess_batch_load_streaming(
    loader,
    concepts: List[str],
    patient_batches,
    total_patients: int,
    batch_size: int,
    verbose: bool = False,
    memory_efficient: bool = False,
    **load_kwargs,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    流式分批加载，不预先物化全部 patient_ids。

    适用于全量 cohort：ID 批次直接从底层存储流出，避免先在 Python 中构造一个超大 ID 列表。
    """
    num_batches = max(1, (total_patients + batch_size - 1) // batch_size)

    if verbose:
        print(f"🔄 流式进程内分批: {total_patients} patients, batch_size={batch_size}, {num_batches} batches")

    buffered_batches: List[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = []
    buffered_mb = 0.0
    estimated_total_mb = 0.0
    representative_batch_mb = 0.0
    spill_dir: Optional[str] = None
    spill_batches = 0

    for batch_num, patient_id_batch in enumerate(patient_batches, start=1):
        id_col = list(patient_id_batch.keys())[0]
        batch_ids = list(patient_id_batch.values())[0]

        if verbose:
            rss = get_rss_mb()
            print(
                f"   📦 Batch {batch_num}/{num_batches}: {len(batch_ids)} patients (RSS: {rss:.0f}MB)...",
                end='',
                flush=True,
            )

        loader.clear_cache()
        batch_result = loader.load_concepts(
            concepts=concepts,
            patient_ids={id_col: batch_ids},
            **load_kwargs,
        )

        if spill_dir is None:
            if representative_batch_mb == 0.0:
                representative_batch_mb = _estimate_result_size_mb(batch_result)
                if representative_batch_mb > 0:
                    estimated_total_mb = representative_batch_mb * num_batches
            batch_result_mb = representative_batch_mb
        else:
            batch_result_mb = 0.0

        if isinstance(batch_result, pd.DataFrame) and len(batch_result) > 0:
            if verbose:
                print(f" ✅ ({len(batch_result)} rows)", end='')
        elif isinstance(batch_result, dict):
            if verbose:
                non_empty = sum(len(df) for df in batch_result.values() if isinstance(df, pd.DataFrame) and len(df) > 0)
                print(f" ✅ ({non_empty} rows / {len(batch_result)} concepts)", end='')
        elif verbose:
            print(" ⚪ (empty)", end='')

        if spill_dir is None and _should_spill_inprocess_batches(
            memory_efficient=memory_efficient,
            num_batches=num_batches,
            estimated_total_mb=estimated_total_mb,
            buffered_mb=buffered_mb + batch_result_mb,
        ):
            spill_dir = tempfile.mkdtemp(prefix='easyicu_streaming_')
            if verbose:
                print(f" 💽 spill→disk[{spill_dir}]", end='')
            for buffered in buffered_batches:
                spill_batches += 1
                _write_batch_result_to_parquet(buffered, os.path.join(spill_dir, f'batch_{spill_batches:04d}'))
            buffered_batches.clear()
            buffered_mb = 0.0
            release_memory(aggressive=True)

        if spill_dir is not None:
            spill_batches += 1
            _write_batch_result_to_parquet(batch_result, os.path.join(spill_dir, f'batch_{spill_batches:04d}'))
            del batch_result
        elif (
            isinstance(batch_result, pd.DataFrame) and not batch_result.empty
        ) or isinstance(batch_result, dict):
            buffered_batches.append(batch_result)
            buffered_mb += batch_result_mb

        freed = release_memory()
        if verbose:
            print(f" [freed {freed}MB, RSS: {get_rss_mb():.0f}MB]")

    if not buffered_batches and spill_dir is None:
        return pd.DataFrame()

    if spill_dir is not None:
        try:
            final = _merge_parquet_batches(spill_dir)
        finally:
            import shutil
            shutil.rmtree(spill_dir, ignore_errors=True)
        release_memory(aggressive=True)
        return final

    final = _merge_buffered_batches(buffered_batches)
    del buffered_batches
    release_memory(aggressive=True)
    return final
