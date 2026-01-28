"""
智能并行配置模块

根据系统内存和CPU核心数自动调整并行加载策略。
用于代码端和Web端统一的并行配置。

Usage:
    from pyricu.parallel_config import get_parallel_config, ParallelConfig
    
    config = get_parallel_config()
    print(f"Max workers: {config.max_workers}")
    print(f"Buckets per batch: {config.buckets_per_batch}")
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """并行配置类"""
    
    # 系统信息
    total_memory_gb: float  # 总内存 (GB)
    available_memory_gb: float  # 可用内存 (GB)
    cpu_count: int  # CPU 核心数
    
    # 并行策略
    max_workers: int  # 最大并行工作线程数
    buckets_per_batch: int  # 每批读取的分桶数
    memory_per_concept_mb: int  # 预估每个概念的内存占用 (MB)
    
    # 优化标志
    use_duckdb_aggregation: bool  # 是否使用DuckDB层聚合
    enable_concept_cache: bool  # 是否启用概念缓存
    
    @property
    def performance_tier(self) -> str:
        """返回性能等级描述"""
        if self.total_memory_gb >= 128:
            return "high-performance"
        elif self.total_memory_gb >= 64:
            return "server"
        elif self.total_memory_gb >= 32:
            return "workstation"
        elif self.total_memory_gb >= 16:
            return "standard"
        else:
            return "limited"


def get_system_memory() -> tuple:
    """获取系统内存信息
    
    Returns:
        (total_memory_gb, available_memory_gb)
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        return total_gb, available_gb
    except ImportError:
        # psutil 未安装，使用系统命令
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
                mem_info = {}
                for line in lines:
                    parts = line.split(':')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip().split()[0]  # 取第一个数字
                        mem_info[key] = int(value)
                
                total_kb = mem_info.get('MemTotal', 16 * 1024 * 1024)
                available_kb = mem_info.get('MemAvailable', 
                                           mem_info.get('MemFree', 8 * 1024 * 1024))
                return total_kb / (1024 ** 2), available_kb / (1024 ** 2)
        except Exception:
            # 默认假设 16GB 内存，8GB 可用
            return 16.0, 8.0


def get_cpu_count() -> int:
    """获取CPU核心数"""
    try:
        import psutil
        return psutil.cpu_count(logical=True) or os.cpu_count() or 4
    except ImportError:
        return os.cpu_count() or 4


def get_parallel_config(
    override_memory_gb: Optional[float] = None,
    override_workers: Optional[int] = None,
) -> ParallelConfig:
    """
    获取智能并行配置
    
    根据系统资源自动计算最优的并行策略。
    
    Args:
        override_memory_gb: 手动指定内存大小 (GB)，用于测试
        override_workers: 手动指定最大工作线程数
        
    Returns:
        ParallelConfig 配置对象
        
    Examples:
        # 自动检测
        config = get_parallel_config()
        
        # 手动指定 (用于测试或限制资源)
        config = get_parallel_config(override_memory_gb=8, override_workers=2)
    """
    # 获取系统信息
    total_mem, available_mem = get_system_memory()
    cpu_count = get_cpu_count()
    
    if override_memory_gb is not None:
        total_mem = override_memory_gb
        available_mem = override_memory_gb * 0.7  # 假设70%可用
    
    # 计算并行策略
    # 基于内存的策略（保守估计，每个并行任务需要约2GB内存）
    memory_based_workers = max(1, int(available_mem / 2))
    
    # 基于CPU的策略（不超过CPU核心数的一半，避免过度竞争）
    cpu_based_workers = max(1, cpu_count // 2)
    
    # 取较小值，确保不会OOM
    max_workers = min(memory_based_workers, cpu_based_workers)
    
    # 🚀 根据内存大小动态调整上限
    # 16GB: 最多8个workers
    # 32GB: 最多16个workers
    # 64GB: 最多32个workers
    # 128GB+: 最多64个workers
    if total_mem >= 128:
        max_workers_limit = 64
    elif total_mem >= 64:
        max_workers_limit = 32
    elif total_mem >= 32:
        max_workers_limit = 16
    else:
        max_workers_limit = 8
    
    max_workers = min(max_workers, max_workers_limit)
    
    if override_workers is not None:
        max_workers = override_workers
    
    # 每批读取的分桶数
    # 16GB: 1个分桶/批
    # 32GB: 2个分桶/批
    # 64GB+: 4个分桶/批
    # 128GB+: 8个分桶/批
    if total_mem >= 128:
        buckets_per_batch = 8
    elif total_mem >= 64:
        buckets_per_batch = 4
    elif total_mem >= 32:
        buckets_per_batch = 2
    else:
        buckets_per_batch = 1
    
    # 每个概念的预估内存占用
    # 基于经验值：MIIV hr 约200MB，AUMC numericitems 约1GB
    if total_mem >= 64:
        memory_per_concept_mb = 500
    elif total_mem >= 32:
        memory_per_concept_mb = 300
    else:
        memory_per_concept_mb = 200
    
    # 是否启用DuckDB层聚合（始终启用，这是关键优化）
    use_duckdb_aggregation = True
    
    # 是否启用概念缓存（内存充足时启用）
    enable_concept_cache = available_mem >= 8
    
    config = ParallelConfig(
        total_memory_gb=total_mem,
        available_memory_gb=available_mem,
        cpu_count=cpu_count,
        max_workers=max_workers,
        buckets_per_batch=buckets_per_batch,
        memory_per_concept_mb=memory_per_concept_mb,
        use_duckdb_aggregation=use_duckdb_aggregation,
        enable_concept_cache=enable_concept_cache,
    )
    
    logger.info(
        f"🔧 并行配置: {config.performance_tier} "
        f"(内存: {total_mem:.1f}GB, CPU: {cpu_count}核, "
        f"workers: {max_workers}, buckets/batch: {buckets_per_batch})"
    )
    
    return config


def get_recommended_batch_size(
    config: Optional[ParallelConfig] = None,
    num_concepts: int = 1,
    database: str = 'miiv',
) -> int:
    """
    获取推荐的患者批处理大小
    
    用于内存受限环境下的分批处理。
    
    Args:
        config: 并行配置，如果为None则自动获取
        num_concepts: 要加载的概念数量
        database: 数据库类型
        
    Returns:
        推荐的 batch_size
    """
    if config is None:
        config = get_parallel_config()
    
    # 基础批大小
    # 根据 AGENTS.md: 12GB 内存上 30000 患者是安全上限
    if config.available_memory_gb >= 32:
        base_batch = 50000
    elif config.available_memory_gb >= 16:
        base_batch = 30000
    elif config.available_memory_gb >= 8:
        base_batch = 10000
    else:
        base_batch = 5000
    
    # 根据概念数量调整
    # 多个概念会增加内存占用
    concept_factor = max(0.3, 1.0 - (num_concepts - 1) * 0.1)
    
    # 某些数据库的数据量更大，需要更小的批次
    db_factors = {
        'aumc': 0.7,  # AUMC 行数多
        'hirid': 0.6,  # HiRID 高频数据
        'miiv': 1.0,
        'eicu': 1.2,  # eICU 相对较小
        'mimic': 1.0,
        'sic': 1.0,
    }
    db_factor = db_factors.get(database, 1.0)
    
    recommended = int(base_batch * concept_factor * db_factor)
    
    # 确保至少1000
    return max(1000, recommended)


# 全局配置缓存
_cached_config: Optional[ParallelConfig] = None


def get_global_config() -> ParallelConfig:
    """获取全局并行配置（带缓存）"""
    global _cached_config
    if _cached_config is None:
        _cached_config = get_parallel_config()
    return _cached_config


def reset_global_config():
    """重置全局配置缓存"""
    global _cached_config
    _cached_config = None
