"""
大规模数据加载优化模块
整合了多种性能优化策略，可以在easyicu内部使用
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
import pyarrow.dataset as ds

logger = logging.getLogger(__name__)

class TablePreloader:
    """
    表数据预加载器
    
    优化策略:
    1. 一次性读取大表，避免重复I/O
    2. 只加载必要的列
    3. 过滤目标患者数据
    4. 缓存在内存中供后续使用
    """
    
    def __init__(self, data_path: Path, enable_preload: bool = True):
        self.data_path = data_path
        self.enable_preload = enable_preload
        self.cache: Dict[str, pd.DataFrame] = {}
        self.stats: Dict[str, Dict] = {}
    
    def preload_for_patients(
        self, 
        patient_ids: List[int],
        tables: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """
        为指定患者预加载表数据
        
        Args:
            patient_ids: 患者ID列表
            tables: 要预加载的表及其列，格式为 {table_name: [columns]}
                   如果为None，使用默认表列表
        """
        if not self.enable_preload:
            logger.info("预加载已禁用")
            return
        
        # 默认预加载配置
        if tables is None:
            tables = self._get_default_tables()
        
        logger.info(f"📦 开始预加载 {len(tables)} 个表...")
        patient_set = set(patient_ids)
        total_start = time.perf_counter()
        
        for table_name, columns in tables.items():
            start_time = time.perf_counter()
            
            try:
                table_path = self.data_path / f"{table_name}.parquet"
                
                # 如果是单文件不存在，尝试分区目录
                if not table_path.exists():
                    table_path = self.data_path / table_name
                
                # 读取数据
                if table_path.is_dir():
                    # 分区表 - 使用pyarrow dataset读取（忽略.fst文件）
                    dataset = ds.dataset(
                        table_path,
                        format='parquet',
                        partitioning=None,
                        exclude_invalid_files=True
                    )
                    df = dataset.to_table(columns=columns).to_pandas()
                else:
                    # 单文件表
                    df = pd.read_parquet(table_path, columns=columns)
                
                # 如果表有stay_id列，只保留目标患者
                if 'stay_id' in df.columns:
                    original_rows = len(df)
                    df = df[df['stay_id'].isin(patient_set)]
                    filtered_rows = len(df)
                    filter_ratio = (1 - filtered_rows/original_rows) * 100 if original_rows > 0 else 0
                else:
                    filtered_rows = len(df)
                    filter_ratio = 0
                
                # 缓存数据
                self.cache[table_name] = df
                
                elapsed = time.perf_counter() - start_time
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                
                self.stats[table_name] = {
                    'rows': filtered_rows,
                    'columns': len(df.columns),
                    'memory_mb': memory_mb,
                    'load_time': elapsed,
                    'filter_ratio': filter_ratio
                }
                
                if filter_ratio > 0:
                    logger.info(
                        f"  ✅ {table_name}: {filtered_rows:,}行 "
                        f"({filter_ratio:.1f}% 过滤), {memory_mb:.1f}MB, {elapsed:.2f}s"
                    )
                else:
                    logger.info(
                        f"  ✅ {table_name}: {filtered_rows:,}行, "
                        f"{memory_mb:.1f}MB, {elapsed:.2f}s"
                    )
                
            except Exception as e:
                logger.warning(f"  ⚠️  {table_name} 跳过: {e}")
                self.cache[table_name] = None
        
        total_time = time.perf_counter() - total_start
        total_memory = sum(s['memory_mb'] for s in self.stats.values())
        total_rows = sum(s['rows'] for s in self.stats.values())
        
        logger.info(
            f"\n📊 预加载完成: {total_rows:,}行, {total_memory:.1f}MB, "
            f"{total_time:.2f}s ({total_time/len(patient_ids)*1000:.1f}ms/患者)"
        )
    
    def get_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """获取预加载的表数据"""
        return self.cache.get(table_name)
    
    def is_preloaded(self, table_name: str) -> bool:
        """检查表是否已预加载"""
        return table_name in self.cache and self.cache[table_name] is not None
    
    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
        self.stats.clear()
    
    @staticmethod
    def _get_default_tables() -> Dict[str, List[str]]:
        """
        获取默认的预加载表配置
        只包含有stay_id的大表，以及必要的字典表
        """
        return {
            'chartevents': ['stay_id', 'charttime', 'itemid', 'valuenum', 'valueuom', 'value'],
            # labevents没有stay_id，需要关联，不预加载
            'outputevents': ['stay_id', 'charttime', 'itemid', 'value'],
            'procedureevents': ['stay_id', 'starttime', 'itemid', 'value'],
            'datetimeevents': ['stay_id', 'charttime', 'itemid', 'value'],
            'icustays': ['stay_id', 'hadm_id', 'subject_id', 'intime', 'outtime', 'los'],
            'd_items': ['itemid', 'label', 'category'],
        }

class BatchProcessor:
    """
    批处理器
    
    优化策略:
    1. 将大量患者分成小批次
    2. 控制内存使用
    3. 支持并行处理
    4. 增量保存结果
    """
    
    def __init__(
        self,
        batch_size: int = 100,
        enable_parallel: bool = True,
        num_workers: Optional[int] = None
    ):
        self.batch_size = batch_size
        self.enable_parallel = enable_parallel
        
        if num_workers is None:
            # 默认使用CPU核心数-1，至少为1
            self.num_workers = max(1, mp.cpu_count() - 1)
        else:
            self.num_workers = num_workers
    
    def create_batches(self, patient_ids: List[int]) -> List[List[int]]:
        """将患者ID列表分成批次"""
        batches = []
        for i in range(0, len(patient_ids), self.batch_size):
            batches.append(patient_ids[i:i+self.batch_size])
        return batches
    
    def process_batches(
        self,
        batches: List[List[int]],
        process_func: callable,
        **kwargs
    ) -> List:
        """
        处理批次
        
        Args:
            batches: 患者ID批次列表
            process_func: 处理函数，签名为 func(batch_ids, batch_idx, **kwargs)
            **kwargs: 传递给process_func的额外参数
        
        Returns:
            处理结果列表
        """
        total_batches = len(batches)
        results = []
        
        logger.info(
            f"🔄 开始批处理: {total_batches}个批次, "
            f"{'并行' if self.enable_parallel and self.num_workers > 1 else '串行'}模式"
        )
        
        if self.enable_parallel and self.num_workers > 1:
            # 并行处理
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {}
                
                for idx, batch in enumerate(batches):
                    future = executor.submit(
                        process_func,
                        batch,
                        idx,
                        total_batches,
                        **kwargs
                    )
                    futures[future] = idx
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        batch_idx = futures[future]
                        logger.error(f"批次 {batch_idx+1}/{total_batches} 失败: {e}")
                        results.append(None)
        else:
            # 串行处理
            for idx, batch in enumerate(batches):
                try:
                    result = process_func(batch, idx, total_batches, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"批次 {idx+1}/{total_batches} 失败: {e}")
                    results.append(None)
        
        # 过滤失败的批次
        results = [r for r in results if r is not None]
        
        logger.info(f"✅ 批处理完成: {len(results)}/{total_batches} 个批次成功")
        
        return results

class PerformanceOptimizer:
    """
    性能优化器主类
    整合预加载、批处理等优化策略
    """
    
    def __init__(
        self,
        data_path: Path,
        enable_preload: bool = True,
        enable_batch: bool = True,
        batch_size: int = 100,
        num_workers: Optional[int] = None
    ):
        self.data_path = data_path
        self.enable_preload = enable_preload
        self.enable_batch = enable_batch
        
        self.preloader = TablePreloader(data_path, enable_preload)
        self.batch_processor = BatchProcessor(batch_size, enable_batch, num_workers)
    
    def optimize_loading(
        self,
        patient_ids: List[int],
        preload_tables: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """
        优化数据加载
        
        Args:
            patient_ids: 患者ID列表
            preload_tables: 要预加载的表配置
        """
        if self.enable_preload and len(patient_ids) >= 100:
            # 只有患者数量足够多时才预加载
            logger.info(f"🚀 启用预加载优化（{len(patient_ids)}名患者）")
            self.preloader.preload_for_patients(patient_ids, preload_tables)
        else:
            logger.info("预加载优化已禁用或患者数量过少")
    
    def get_preloaded_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """获取预加载的表数据"""
        return self.preloader.get_table(table_name)
    
    def is_preloaded(self, table_name: str) -> bool:
        """检查表是否已预加载"""
        return self.preloader.is_preloaded(table_name)
    
    def create_batches(self, patient_ids: List[int]) -> List[List[int]]:
        """创建批次"""
        return self.batch_processor.create_batches(patient_ids)
    
    def process_batches(self, batches: List[List[int]], process_func: callable, **kwargs) -> List:
        """处理批次"""
        return self.batch_processor.process_batches(batches, process_func, **kwargs)
    
    def clear_cache(self):
        """清除缓存"""
        self.preloader.clear_cache()

# 全局优化器实例
_global_optimizer: Optional[PerformanceOptimizer] = None

def get_optimizer(
    data_path: Path,
    enable_preload: bool = True,
    enable_batch: bool = True,
    batch_size: int = 100,
    num_workers: Optional[int] = None
) -> PerformanceOptimizer:
    """获取或创建全局优化器实例"""
    global _global_optimizer
    
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer(
            data_path,
            enable_preload,
            enable_batch,
            batch_size,
            num_workers
        )
    
    return _global_optimizer

def enable_performance_optimization(
    data_path: Path,
    patient_ids: List[int],
    preload_tables: Optional[Dict[str, List[str]]] = None,
    batch_size: int = 100,
    num_workers: Optional[int] = None
) -> PerformanceOptimizer:
    """
    启用性能优化
    
    在加载大量患者数据前调用此函数，可以显著提升性能
    
    Args:
        data_path: 数据路径
        patient_ids: 患者ID列表
        preload_tables: 要预加载的表配置
        batch_size: 批次大小
        num_workers: 并行worker数量
    
    Returns:
        优化器实例
    
    Example:
        >>> from easyicu.performance import enable_performance_optimization
        >>> from easyicu import load_concepts
        >>>
        >>> # 启用优化
        >>> optimizer = enable_performance_optimization(
        ...     data_path=Path("/path/to/data"),
        ...     patient_ids=list(range(1000)),  # 1000名患者
        ...     batch_size=100,
        ...     num_workers=8
        ... )
        >>>
        >>> # 加载数据（自动使用优化）
        >>> sofa = load_concepts('sofa', patient_ids=list(range(1000)))
    """
    optimizer = get_optimizer(
        data_path,
        enable_preload=True,
        enable_batch=True,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    optimizer.optimize_loading(patient_ids, preload_tables)
    
    return optimizer
