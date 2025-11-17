"""
优化的数据加载器

基于列裁剪、itemid过滤和患者过滤的高效数据加载系统
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any
import pandas as pd

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    logging.warning("PyArrow不可用，将使用pandas加载（较慢）")

from .datasource import ICUDataSource, FilterSpec, FilterOp
from .config import DataSourceConfig

logger = logging.getLogger(__name__)


class OptimizedICUDataSource(ICUDataSource):
    """优化的ICU数据源，支持列裁剪和智能过滤"""

    def __init__(self, config: DataSourceConfig, base_path: Optional[Path] = None,
                 enable_column_pruning: bool = True, enable_itemid_filtering: bool = True):
        """
        初始化优化数据源

        Args:
            config: 数据源配置
            base_path: 数据基础路径
            enable_column_pruning: 启用列裁剪
            enable_itemid_filtering: 启用itemid过滤
        """
        super().__init__(config=config, base_path=base_path)
        self.enable_column_pruning = enable_column_pruning
        self.enable_itemid_filtering = enable_itemid_filtering

        # SOFA相关的itemid映射
        self.sofa_itemid_mapping = self._init_sofa_itemid_mapping()

        # 列需求映射
        self.column_requirements = self._init_column_requirements()

    def _init_sofa_itemid_mapping(self) -> Dict[str, Dict[str, List[int]]]:
        """初始化SOFA组件的itemid映射"""
        return {
            'sofa_resp': {
                'chartevents': [
                    50821, 50816, 50817, 50818, 50819, 50820, 223835,  # 血气分析
                    220045, 220181, 223761,                          # HR, MAP, SpO2
                    223762,                                          # 温度
                    220052, 225312, 52, 443, 456, 6072              # 其他血压相关
                ],
                'labevents': [
                    50821, 50816, 50817, 50818, 50819, 50820, 223835   # 血气分析
                ]
            },
            'sofa_coag': {
                'labevents': [51265]  # 血小板
            },
            'sofa_liver': {
                'labevents': [50885]  # 胆红素
            },
            'sofa_cardio': {
                'chartevents': [
                    220052, 220181, 225312, 52, 443, 456, 6072,       # MAP, 血压
                    220045                                          # HR
                ],
                'inputevents': [
                    221906, 222315, 221289, 221662, 30131, 221749,  # 血管活性药物
                    226208, 226209, 226210, 226211, 226212, 226213   # 氧疗相关
                ]
            },
            'sofa_cns': {
                'chartevents': [198, 220739, 220181]  # GCS评分
            },
            'sofa_renal': {
                'labevents': [50912],                    # 肌酐
                'outputevents': [226559, 226558, 226560]  # 尿量
            }
        }

    def _init_column_requirements(self) -> Dict[str, Set[str]]:
        """初始化各表的基本列需求"""
        return {
            'chartevents': {'stay_id', 'charttime', 'itemid', 'valuenum'},
            'labevents': {'stay_id', 'charttime', 'itemid', 'valuenum'},
            'inputevents': {'stay_id', 'starttime', 'endtime', 'itemid', 'amount'},
            'outputevents': {'stay_id', 'charttime', 'itemid', 'value'},
            'procedureevents': {'stay_id', 'charttime', 'itemid'},
            'icustays': {'stay_id', 'subject_id', 'intime', 'outtime'},
            'patients': {'subject_id', 'gender', 'anchor_age'}
        }

    def get_required_columns(self, table_name: str, itemids: Optional[List[int]] = None) -> List[str]:
        """
        获取表所需的列

        Args:
            table_name: 表名
            itemids: itemid列表（可选）

        Returns:
            需要的列列表
        """
        if not self.enable_column_pruning:
            return None  # 返回None表示读取所有列

        # 基础列需求
        required_columns = self.column_requirements.get(table_name, set()).copy()

        # 根据itemid添加特定列
        if itemids and table_name == 'inputevents':
            required_columns.add('rate')      # 输液速率
            required_columns.add('rateuom')   # 速率单位

        return sorted(list(required_columns))

    def get_relevant_itemids(self, table_name: str, concept_name: Optional[str] = None) -> List[int]:
        """
        获取表相关的itemid

        Args:
            table_name: 表名
            concept_name: 概念名称（可选）

        Returns:
            相关的itemid列表
        """
        if not self.enable_itemid_filtering:
            return None  # 返回None表示不过滤itemid

        if concept_name and concept_name in self.sofa_itemid_mapping:
            return self.sofa_itemid_mapping[concept_name].get(table_name, [])

        # 如果没有指定概念，返回所有SOFA相关的itemid
        all_itemids = set()
        for component_mapping in self.sofa_itemid_mapping.values():
            all_itemids.update(component_mapping.get(table_name, []))

        return sorted(list(all_itemids)) if all_itemids else None

    def _load_raw_frame_optimized(
        self,
        table_name: str,
        columns: Optional[Iterable[str]] = None,
        patient_ids_filter: Optional[FilterSpec] = None,
        concept_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        优化的原始数据帧加载

        Args:
            table_name: 表名
            columns: 列列表
            patient_ids_filter: 患者ID过滤器
            concept_name: 概念名称

        Returns:
            加载的数据帧
        """
        start_time = time.time()

        # 获取优化的列需求
        required_columns = self.get_required_columns(table_name)
        if required_columns and columns:
            # 合并用户指定的列和必需的列
            columns = list(set(columns) | set(required_columns))
        elif required_columns:
            columns = required_columns

        # 获取相关的itemid
        relevant_itemids = self.get_relevant_itemids(table_name, concept_name)

        # 获取文件路径
        file_path = self._resolve_loader_from_disk(table_name)
        if not file_path:
            return self._handle_missing_table(table_name, columns)

        try:
            # 尝试使用PyArrow优化加载
            if PYARROW_AVAILABLE:
                df = self._load_with_pyarrow(file_path, columns, patient_ids_filter, relevant_itemids)
            else:
                df = self._load_with_pandas(file_path, columns, patient_ids_filter, relevant_itemids)

            load_time = time.time() - start_time
            logger.info(f"✅ 优化加载 {table_name}: {len(df):,}行, {load_time:.2f}秒")

            return df

        except Exception as e:
            logger.error(f"❌ 加载失败 {table_name}: {e}")
            # 回退到原始方法
            return super()._load_raw_frame(table_name, columns, patient_ids_filter)

    def _load_with_pyarrow(
        self,
        file_path: Path,
        columns: Optional[List[str]],
        patient_ids_filter: Optional[FilterSpec],
        relevant_itemids: Optional[List[int]]
    ) -> pd.DataFrame:
        """使用PyArrow优化加载"""
        # 构建过滤器
        filters = []

        # 患者ID过滤
        if patient_ids_filter and patient_ids_filter.op == FilterOp.IN:
            filters.append((patient_ids_filter.column, 'in', patient_ids_filter.value))

        # Itemid过滤
        if relevant_itemids:
            filters.append(('itemid', 'in', relevant_itemids))

        # 读取数据
        dataset = pq.ParquetDataset(file_path)

        # 尝试使用过滤器，如果不支持则回退到pandas过滤
        try:
            table = dataset.read(
                columns=columns,
                filters=filters if filters else None
            )
            df = table.to_pandas()
        except TypeError as e:
            if 'filters' in str(e):
                # 过滤器不支持，使用pandas方式读取后过滤
                logger.warning(f"PyArrow过滤器不支持，使用pandas过滤: {e}")
                table = dataset.read(columns=columns)
                df = table.to_pandas()

                # 手动应用过滤器
                for filter_col, filter_op, filter_val in filters:
                    if filter_op == 'in':
                        df = df[df[filter_col].isin(filter_val)]
            else:
                raise e

        return df

    def _load_with_pandas(
        self,
        file_path: Path,
        columns: Optional[List[str]],
        patient_ids_filter: Optional[FilterSpec],
        relevant_itemids: Optional[List[int]]
    ) -> pd.DataFrame:
        """使用pandas加载（回退方案）"""
        # 读取数据
        df = pd.read_parquet(file_path, columns=columns, engine='pyarrow')

        # 应用过滤器
        if patient_ids_filter and patient_ids_filter.op == FilterOp.IN:
            df = df[df[patient_ids_filter.column].isin(patient_ids_filter.value)]

        if relevant_itemids:
            df = df[df['itemid'].isin(relevant_itemids)]

        return df

    def _handle_missing_table(self, table_name: str, columns: Optional[List[str]]) -> pd.DataFrame:
        """处理缺失的表"""
        if self.config.name == 'miiv' and table_name in self.config.tables:
            # 返回空DataFrame，保持列结构
            return pd.DataFrame(columns=columns or ['index'])
        else:
            raise KeyError(f"Table not found: {table_name}")

    def load_table_optimized(
        self,
        table_name: str,
        *,
        columns: Optional[Iterable[str]] = None,
        filters: Optional[Iterable[FilterSpec]] = None,
        concept_name: Optional[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        优化的表加载方法

        Args:
            table_name: 表名
            columns: 列列表
            filters: 过滤器列表
            concept_name: 概念名称
            verbose: 是否显示详细信息

        Returns:
            加载的数据表
        """
        if verbose:
            logger.info(f"🔍 开始优化加载表: {table_name}")

        # 提取患者ID过滤器
        patient_ids_filter = None
        if filters:
            id_columns = ['stay_id', 'subject_id', 'icustay_id', 'hadm_id',
                         'patientunitstayid', 'admissionid', 'patientid']
            for spec in filters:
                if spec.op == FilterOp.IN and spec.column in id_columns:
                    patient_ids_filter = spec
                    break

        # 使用优化的加载方法
        frame = self._load_raw_frame_optimized(
            table_name=table_name,
            columns=columns,
            patient_ids_filter=patient_ids_filter,
            concept_name=concept_name
        )

        # 应用其他过滤器
        if filters:
            for spec in filters:
                frame = spec.apply(frame)

        return frame


class OptimizedLoaderFactory:
    """优化加载器工厂"""

    @staticmethod
    def create_optimized_datasource(
        database: str,
        data_path: Optional[Path] = None,
        **kwargs
    ) -> OptimizedICUDataSource:
        """
        创建优化的数据源

        Args:
            database: 数据库名称
            data_path: 数据路径
            **kwargs: 其他参数

        Returns:
            优化的数据源实例
        """
        from .config import load_data_sources

        registry = load_data_sources()
        config = registry.get(database)
        if not config:
            raise ValueError(f"Unknown database: {database}")

        return OptimizedICUDataSource(config, data_path, **kwargs)

    @staticmethod
    def benchmark_loading(
        database: str,
        data_path: Path,
        table_name: str,
        patient_ids: List[int],
        concept_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        对比优化前后的加载性能

        Args:
            database: 数据库名称
            data_path: 数据路径
            table_name: 表名
            patient_ids: 患者ID列表
            concept_name: 概念名称

        Returns:
            性能对比结果
        """
        logger.info(f"🏁 开始性能基准测试: {table_name}")

        # 测试传统加载
        traditional_start = time.time()
        try:
            traditional_source = ICUDataSource(
                config=load_data_sources().get(database),
                base_path=data_path
            )
            traditional_df = traditional_source.load_table(
                table_name,
                filters=[FilterSpec('stay_id', FilterOp.IN, patient_ids)]
            )
            traditional_time = time.time() - traditional_start
            traditional_size = len(traditional_df)
        except Exception as e:
            logger.error(f"传统加载失败: {e}")
            traditional_time = float('inf')
            traditional_size = 0
            traditional_df = pd.DataFrame()

        # 测试优化加载
        optimized_start = time.time()
        try:
            optimized_source = OptimizedICUDataSource(
                config=load_data_sources().get(database),
                base_path=data_path
            )
            optimized_df = optimized_source.load_table_optimized(
                table_name,
                filters=[FilterSpec('stay_id', FilterOp.IN, patient_ids)],
                concept_name=concept_name
            )
            optimized_time = time.time() - optimized_start
            optimized_size = len(optimized_df)
        except Exception as e:
            logger.error(f"优化加载失败: {e}")
            optimized_time = float('inf')
            optimized_size = 0
            optimized_df = pd.DataFrame()

        # 计算性能提升
        speedup = traditional_time / optimized_time if optimized_time > 0 else float('inf')
        size_reduction = (traditional_size - optimized_size) / traditional_size if traditional_size > 0 else 0

        result = {
            'table_name': table_name,
            'concept_name': concept_name,
            'patient_count': len(patient_ids),
            'traditional_time': traditional_time,
            'optimized_time': optimized_time,
            'speedup': speedup,
            'traditional_rows': traditional_size,
            'optimized_rows': optimized_size,
            'size_reduction_percent': size_reduction * 100,
            'memory_saving_mb': (traditional_size - optimized_size) * 0.1  # 估算
        }

        logger.info(f"📊 基准测试结果 {table_name}:")
        logger.info(f"   时间: {traditional_time:.2f}s → {optimized_time:.2f}s ({speedup:.1f}x)")
        logger.info(f"   行数: {traditional_size:,} → {optimized_size:,} ({size_reduction*100:.1f}% 减少)")

        return result


# 全局优化加载器实例
_optimized_sources = {}


def get_optimized_datasource(
    database: str,
    data_path: Optional[Path] = None,
    **kwargs
) -> OptimizedICUDataSource:
    """
    获取优化的数据源实例（单例模式）

    Args:
        database: 数据库名称
        data_path: 数据路径
        **kwargs: 其他参数

    Returns:
        优化的数据源实例
    """
    key = (database, str(data_path), frozenset(kwargs.items()))

    if key not in _optimized_sources:
        _optimized_sources[key] = OptimizedLoaderFactory.create_optimized_datasource(
            database, data_path, **kwargs
        )

    return _optimized_sources[key]