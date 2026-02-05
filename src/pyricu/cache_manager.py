"""
pyricu 缓存管理工具
提供统一的缓存清理和管理功能
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any
import logging

from .project_config import AUTO_CLEAR_CACHE, CACHE_DIR

logger = logging.getLogger(__name__)

class CacheManager:
    """全局缓存管理器"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._cache_dirs = []
            self._memory_caches = []
            self._initialized = True
            self._setup_default_cache_dirs()

    def _setup_default_cache_dirs(self):
        """设置默认的缓存目录"""
        # 项目缓存目录
        if CACHE_DIR.exists():
            self._cache_dirs.append(CACHE_DIR)

        # 用户主目录下的pyricu缓存
        home_cache = Path.home() / ".pyricu_cache"
        if home_cache.exists():
            self._cache_dirs.append(home_cache)

        # 临时目录中的pyricu缓存
        temp_cache = Path(tempfile.gettempdir()) / "pyricu_cache"
        if temp_cache.exists():
            self._cache_dirs.append(temp_cache)

        # 系统缓存目录
        if os.name == 'posix':  # Unix/Linux/macOS
            system_cache = Path("/tmp") / "pyricu_cache"
            if system_cache.exists():
                self._cache_dirs.append(system_cache)

    def register_memory_cache(self, cache_obj: Any):
        """注册内存缓存对象，需要实现clear()方法"""
        self._memory_caches.append(cache_obj)

    def clear_disk_cache(self) -> Dict[str, bool]:
        """清除所有磁盘缓存"""
        results = {}

        for cache_dir in self._cache_dirs:
            try:
                if cache_dir.exists():
                    # 删除整个目录及其内容
                    shutil.rmtree(cache_dir)
                    # 重新创建空目录
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    results[str(cache_dir)] = True
                    logger.info(f"✅ 已清除磁盘缓存: {cache_dir}")
                else:
                    results[str(cache_dir)] = True  # 不存在也算清除成功
            except Exception as e:
                results[str(cache_dir)] = False
                logger.warning(f"⚠️  清除磁盘缓存失败: {cache_dir} - {e}")

        return results

    def clear_memory_cache(self) -> Dict[str, bool]:
        """清除所有注册的内存缓存"""
        results = {}

        for i, cache_obj in enumerate(self._memory_caches):
            try:
                if hasattr(cache_obj, 'clear'):
                    cache_obj.clear()
                    results[f"memory_cache_{i}"] = True
                    logger.info(f"✅ 已清除内存缓存: {type(cache_obj).__name__}")
                else:
                    results[f"memory_cache_{i}"] = False
                    logger.warning(f"⚠️  缓存对象没有clear方法: {type(cache_obj).__name__}")
            except Exception as e:
                results[f"memory_cache_{i}"] = False
                logger.warning(f"⚠️  清除内存缓存失败: {type(cache_obj).__name__} - {e}")

        return results

    def clear_all_cache(self) -> Dict[str, Any]:
        """清除所有缓存（磁盘+内存）"""
        logger.info("🧹 开始清除所有pyricu缓存...")

        disk_results = self.clear_disk_cache()
        memory_results = self.clear_memory_cache()

        summary = {
            'disk_cache': disk_results,
            'memory_cache': memory_results,
            'total_disk_dirs': len(disk_results),
            'successful_disk_clears': sum(disk_results.values()),
            'total_memory_caches': len(memory_results),
            'successful_memory_clears': sum(memory_results.values())
        }

        success_count = summary['successful_disk_clears'] + summary['successful_memory_clears']
        total_count = summary['total_disk_dirs'] + summary['total_memory_caches']

        if success_count == total_count:
            logger.info(f"🎉 缓存清除完成: {success_count}/{total_count} 个缓存已清除")
        else:
            logger.warning(f"⚠️  部分缓存清除失败: {success_count}/{total_count} 个缓存已清除")

        return summary

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        info = {
            'disk_cache_dirs': [],
            'memory_cache_count': len(self._memory_caches),
            'auto_clear_enabled': AUTO_CLEAR_CACHE
        }

        for cache_dir in self._cache_dirs:
            if cache_dir.exists():
                size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                file_count = len(list(cache_dir.rglob('*')))
                info['disk_cache_dirs'].append({
                    'path': str(cache_dir),
                    'size_mb': round(size / (1024 * 1024), 2),
                    'file_count': file_count
                })
            else:
                info['disk_cache_dirs'].append({
                    'path': str(cache_dir),
                    'size_mb': 0,
                    'file_count': 0,
                    'exists': False
                })

        return info

# 全局缓存管理器实例
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager

def auto_clear_cache_if_enabled():
    """如果启用了自动清除，则清除缓存"""
    if AUTO_CLEAR_CACHE:
        logger.info("🔄 自动缓存清除已启用，正在清理缓存...")
        cache_manager = get_cache_manager()
        return cache_manager.clear_all_cache()
    else:
        logger.info("ℹ️  自动缓存清除已禁用")
        return None

def clear_pyricu_cache():
    """手动清除pyricu缓存的便捷函数"""
    # 1. 清除 CacheManager 管理的缓存（磁盘 + 注册的内存缓存）
    cache_manager = get_cache_manager()
    result = cache_manager.clear_all_cache()
    
    # 2. 清除全局加载器（重要：否则患者ID可能被缓存）
    try:
        from .api import clear_global_loader
        clear_global_loader()
        logger.info("✅ 已清除全局加载器")
    except ImportError:
        pass
    
    return result

def get_cache_status():
    """获取缓存状态的便捷函数"""
    cache_manager = get_cache_manager()
    return cache_manager.get_cache_info()

# 在模块导入时自动执行（如果启用）
def _initialize_cache_manager():
    """初始化缓存管理器"""
    try:
        cache_manager = get_cache_manager()
        if AUTO_CLEAR_CACHE:
            cache_manager.clear_all_cache()
    except Exception as e:
        logger.warning(f"初始化缓存管理器时出错: {e}")