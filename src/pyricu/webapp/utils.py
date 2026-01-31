"""PyRICU Webapp 工具函数模块。

提取自 app.py，包含通用工具函数。
"""

import re
import numpy as np
from typing import Dict, Any


def get_system_resources() -> Dict[str, Any]:
    """检测系统硬件资源。
    
    使用统一的 parallel_config 模块，确保代码端和 Web 端配置一致。
    
    Returns:
        dict: 包含 cpu_count, memory_gb, recommended_workers, recommended_backend
    """
    try:
        from ..parallel_config import get_global_config
        config = get_global_config()
        
        # 根据配置选择后端
        if config.cpu_count >= 16 and config.total_memory_gb >= 32:
            recommended_backend = "loky"
        else:
            recommended_backend = "thread"
        
        return {
            'cpu_count': config.cpu_count,
            'total_memory_gb': round(config.total_memory_gb, 1),
            'available_memory_gb': round(config.available_memory_gb, 1),
            'recommended_workers': config.max_workers,
            'recommended_backend': recommended_backend,
            'performance_tier': config.performance_tier,
            'buckets_per_batch': config.buckets_per_batch,
        }
    except ImportError:
        # Fallback: 直接检测（兼容旧版本）
        import os
        try:
            import psutil
            mem_info = psutil.virtual_memory()
            total_memory_gb = mem_info.total / (1024 ** 3)
            available_memory_gb = mem_info.available / (1024 ** 3)
        except Exception:
            total_memory_gb = 8
            available_memory_gb = 4
        
        cpu_count = os.cpu_count() or 4
        max_workers_by_memory = int(available_memory_gb / 2)
        max_workers_by_cpu = int(cpu_count * 0.75)
        recommended_workers = min(max_workers_by_memory, max_workers_by_cpu, 64)
        recommended_workers = max(recommended_workers, 1)
        
        if cpu_count >= 16 and total_memory_gb >= 32:
            recommended_backend = "loky"
        else:
            recommended_backend = "thread"
        
        return {
            'cpu_count': cpu_count,
            'total_memory_gb': round(total_memory_gb, 1),
            'available_memory_gb': round(available_memory_gb, 1),
            'recommended_workers': recommended_workers,
            'recommended_backend': recommended_backend,
        }


def get_optimal_parallel_config(num_patients: int = None, task_type: str = 'load'):
    """根据系统资源和任务规模返回最优的并行配置。
    
    Args:
        num_patients: 要处理的患者数量，None 表示未知/全量
        task_type: 任务类型 ('load', 'export', 'preview')
    
    Returns:
        tuple: (parallel_workers, parallel_backend)
    """
    resources = get_system_resources()
    base_workers = resources['recommended_workers']
    backend = resources['recommended_backend']
    
    # 根据任务类型调整
    if task_type == 'preview':
        # 预览只需少量数据，不需要太多并行
        workers = min(base_workers, 4)
        backend = "thread"  # 预览用线程更快启动
    elif task_type == 'load':
        # 数据加载根据患者数量调整
        if num_patients is None or num_patients >= 50000:
            workers = base_workers  # 全量使用推荐配置
        elif num_patients >= 10000:
            workers = min(base_workers, max(8, base_workers // 2))
        elif num_patients >= 2000:
            workers = min(base_workers, 4)
        else:
            workers = 1  # 少量患者不需要并行
    elif task_type == 'export':
        # 导出任务可以使用更多资源
        workers = base_workers
    else:
        workers = min(base_workers, 8)
    
    # Streamlit webapp 环境下，线程通常更安全
    # 只有在明确高配置环境下才使用进程池
    if backend == "loky" and task_type != 'export':
        backend = "thread"  # webapp 中优先使用线程
    
    return workers, backend


def strip_emoji(text: str) -> str:
    """移除字符串中的emoji字符，用于CSV导出等场景防止乱码。
    
    Args:
        text: 输入字符串
        
    Returns:
        移除emoji后的字符串
    """
    # 匹配更全面的emoji范围
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000026FF"  # Misc symbols (includes 🧪 etc)
        "\U00002B50-\U00002B55"  # stars
        "\U0001F004-\U0001F0CF"  # mahjong
        "\U0000203C-\U00003299"  # misc symbols
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text).strip()


def safe_format_number(val, decimals: int = 0) -> str:
    """安全地格式化数值，处理非数值类型（如字符串、NaN等）。
    
    Args:
        val: 要格式化的值
        decimals: 小数位数
        
    Returns:
        格式化后的字符串
    """
    # 处理 None 和 NaN
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    
    # 如果是字符串类型，直接返回
    if isinstance(val, (str, np.str_)):
        return str(val)
    
    # 尝试数值格式化
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def find_database_path(root: str, db_name: str) -> str:
    """在根目录下查找数据库目录。
    
    Args:
        root: 根目录路径
        db_name: 数据库名称
        
    Returns:
        数据库目录路径，未找到返回空字符串
    """
    from pathlib import Path
    
    # 数据库名称到目录名的映射
    db_dir_map = {
        'miiv': ['mimiciv', 'mimic-iv', 'mimic_iv'],
        'eicu': ['eicu', 'eicu-crd'],
        'aumc': ['aumc', 'amsterdamumc', 'amsterdamumcdb'],
        'hirid': ['hirid'],
        'mimic': ['mimiciii', 'mimic-iii', 'mimic_iii'],
        'sic': ['sicdb', 'sic'],
    }
    
    root_path = Path(root)
    if not root_path.exists():
        return ""
    
    # 查找匹配的目录
    candidates = db_dir_map.get(db_name, [db_name])
    for subdir in root_path.iterdir():
        if subdir.is_dir():
            subdir_lower = subdir.name.lower()
            for candidate in candidates:
                if candidate.lower() in subdir_lower:
                    # 查找版本子目录
                    version_dirs = sorted([
                        d for d in subdir.iterdir() 
                        if d.is_dir() and d.name[0].isdigit()
                    ], reverse=True)
                    if version_dirs:
                        return str(version_dirs[0])
                    return str(subdir)
    return ""


def generate_cohort_prefix() -> str:
    """生成队列导出的唯一前缀。
    
    Returns:
        格式为 cohort_YYYYMMDD_HHMMSS 的字符串
    """
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"cohort_{timestamp}"
