"""PyRICU Web 应用工具函数。

包含系统资源检测、并行配置等实用函数。
"""

import os
import psutil
from typing import Dict, Tuple, Optional


def get_system_resources() -> Dict:
    """获取系统资源信息，用于智能配置并行参数。
    
    Returns:
        包含 CPU、内存信息和推荐配置的字典
    """
    cpu_count = os.cpu_count() or 4
    
    try:
        mem = psutil.virtual_memory()
        total_memory_gb = round(mem.total / (1024**3), 1)
        available_memory_gb = round(mem.available / (1024**3), 1)
    except Exception:
        total_memory_gb = 16
        available_memory_gb = 8
    
    # 根据系统资源推荐并行配置
    if total_memory_gb >= 64 and cpu_count >= 16:
        recommended_workers = min(8, cpu_count // 2)
        recommended_backend = 'loky'
        performance_tier = 'high-performance'
    elif total_memory_gb >= 32 and cpu_count >= 8:
        recommended_workers = min(6, cpu_count // 2)
        recommended_backend = 'loky'
        performance_tier = 'server'
    elif total_memory_gb >= 16 and cpu_count >= 4:
        recommended_workers = min(4, cpu_count)
        recommended_backend = 'threading'
        performance_tier = 'workstation'
    elif total_memory_gb >= 8:
        recommended_workers = min(2, cpu_count)
        recommended_backend = 'threading'
        performance_tier = 'standard'
    else:
        recommended_workers = 1
        recommended_backend = 'sequential'
        performance_tier = 'limited'
    
    return {
        'cpu_count': cpu_count,
        'total_memory_gb': total_memory_gb,
        'available_memory_gb': available_memory_gb,
        'recommended_workers': recommended_workers,
        'recommended_backend': recommended_backend,
        'performance_tier': performance_tier,
    }


def get_optimal_parallel_config(
    num_patients: Optional[int] = None,
    task_type: str = 'export'
) -> Tuple[int, str]:
    """根据任务类型和患者数量获取最优并行配置。
    
    Args:
        num_patients: 患者数量（可选）
        task_type: 任务类型 ('export', 'load', 'viz')
    
    Returns:
        (workers, backend) 元组
    """
    resources = get_system_resources()
    base_workers = resources['recommended_workers']
    base_backend = resources['recommended_backend']
    
    # 根据任务类型调整
    if task_type == 'viz':
        # 可视化任务需要更快的响应，减少并行
        workers = min(2, base_workers)
        backend = 'threading'
    elif task_type == 'load':
        # 数据加载可以使用更多并行
        workers = base_workers
        backend = base_backend
    else:  # export
        # 导出任务可以使用完整并行
        workers = base_workers
        backend = base_backend
    
    # 根据患者数量调整
    if num_patients is not None:
        if num_patients < 100:
            workers = min(2, workers)
        elif num_patients < 1000:
            workers = min(4, workers)
        # 大量患者保持完整并行
    
    return workers, backend


def format_time_delta(seconds: float) -> str:
    """格式化时间差为人类可读字符串。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def format_memory_size(bytes_size: int) -> str:
    """格式化内存大小为人类可读字符串。"""
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024**2:
        return f"{bytes_size/1024:.1f}KB"
    elif bytes_size < 1024**3:
        return f"{bytes_size/1024**2:.1f}MB"
    else:
        return f"{bytes_size/1024**3:.2f}GB"
