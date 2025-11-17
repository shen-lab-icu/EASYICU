"""
优化的API接口

提供高性能的概念加载功能，集成列裁剪和智能过滤
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import pandas as pd

from .optimized_loader import get_optimized_datasource, OptimizedLoaderFactory
from .concept import ConceptDictionary, ConceptResolver
from .base import BaseICULoader
from .cache_manager import get_cache_manager
from .datasource import FilterSpec, FilterOp

logger = logging.getLogger(__name__)


class OptimizedConceptLoader(BaseICULoader):
    """优化的概念加载器"""

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        database: Optional[str] = None,
        dict_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        use_sofa2: bool = False,
        verbose: bool = False,
        enable_optimizations: bool = True,
        benchmark_mode: bool = False
    ):
        """
        初始化优化概念加载器

        Args:
            data_path: 数据路径
            database: 数据库类型
            dict_path: 字典路径
            use_sofa2: 是否使用SOFA2字典
            verbose: 详细日志
            enable_optimizations: 启用优化
            benchmark_mode: 基准测试模式
        """
        super().__init__(data_path, database, dict_path, use_sofa2, verbose)

        self.enable_optimizations = enable_optimizations
        self.benchmark_mode = benchmark_mode

        if enable_optimizations:
            # 替换数据源为优化版本
            self.datasource = get_optimized_datasource(
                database=self.database,
                data_path=self.data_path,
                enable_column_pruning=True,
                enable_itemid_filtering=True
            )

        self.benchmark_results = []

    def load_concepts_optimized(
        self,
        concepts: Union[str, List[str]],
        patient_ids: Optional[Union[List, Dict]] = None,
        interval: Optional[Union[str, pd.Timedelta]] = None,
        win_length: Optional[Union[str, pd.Timedelta]] = None,
        aggregate: Optional[Union[str, Dict]] = None,
        keep_components: bool = False,
        use_sofa2: bool = False,
        merge: bool = True,
        verbose: bool = False,
        benchmark: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        优化的概念加载方法

        Args:
            concepts: 概念名称或列表
            patient_ids: 患者ID
            interval: 时间间隔
            win_length: 窗口长度
            aggregate: 聚合方式
            keep_components: 保留组件
            use_sofa2: 使用SOFA2
            merge: 是否合并结果
            verbose: 详细日志
            benchmark: 是否进行基准测试
            **kwargs: 其他参数

        Returns:
            加载的概念数据
        """
        if not self.enable_optimizations:
            # 回退到原始方法
            return self.load_concepts(
                concepts=concepts,
                patient_ids=patient_ids,
                interval=interval,
                win_length=win_length,
                aggregate=aggregate,
                keep_components=keep_components,
                use_sofa2=use_sofa2,
                merge=merge,
                verbose=verbose,
                **kwargs
            )

        start_time = time.time()

        if verbose:
            logger.info(f"🚀 开始优化加载概念: {concepts}")

        # 标准化概念列表
        if isinstance(concepts, str):
            concepts = [concepts]

        # 准备患者ID
        if patient_ids is None:
            patient_ids_list = None
        elif isinstance(patient_ids, dict):
            patient_ids_list = list(patient_ids.values())[0] if patient_ids else None
        else:
            patient_ids_list = patient_ids

        # 加载概念数据
        results = {}
        for concept in concepts:
            if benchmark:
                concept_start = time.time()

            try:
                # 使用优化的数据源加载
                concept_data = self._load_single_concept_optimized(
                    concept=concept,
                    patient_ids=patient_ids_list,
                    interval=interval,
                    win_length=win_length,
                    aggregate=aggregate,
                    use_sofa2=use_sofa2
                )

                results[concept] = concept_data

                if benchmark:
                    concept_time = time.time() - concept_start
                    self.benchmark_results.append({
                        'concept': concept,
                        'load_time': concept_time,
                        'rows': len(concept_data) if hasattr(concept_data, '__len__') else 0,
                        'patient_count': len(patient_ids_list) if patient_ids_list else 0
                    })

            except Exception as e:
                logger.error(f"❌ 加载概念失败 {concept}: {e}")
                results[concept] = pd.DataFrame()

        # 合并结果
        if merge and len(results) > 1:
            # 简单合并所有结果
            all_dataframes = []
            for concept_name, df in results.items():
                if not df.empty:
                    # 添加概念名列以区分不同概念
                    df_copy = df.copy()
                    df_copy['concept'] = concept_name
                    all_dataframes.append(df_copy)

            if all_dataframes:
                final_result = pd.concat(all_dataframes, ignore_index=True)
            else:
                final_result = pd.DataFrame()
        elif merge and len(results) == 1:
            final_result = list(results.values())[0]
        else:
            final_result = results

        total_time = time.time() - start_time
        if verbose:
            logger.info(f"✅ 优化加载完成: {total_time:.2f}秒")

        return final_result

    def _load_single_concept_optimized(
        self,
        concept: str,
        patient_ids: Optional[List],
        interval: Optional[Union[str, pd.Timedelta]],
        win_length: Optional[Union[str, pd.Timedelta]],
        aggregate: Optional[Union[str, Dict]],
        use_sofa2: bool
    ) -> pd.DataFrame:
        """
        加载单个优化概念

        Args:
            concept: 概念名称
            patient_ids: 患者ID列表
            interval: 时间间隔
            win_length: 窗口长度
            aggregate: 聚合方式
            use_sofa2: 使用SOFA2

        Returns:
            概念数据
        """
        # 获取概念定义
        concept_info = self.concept_resolver.dictionary.get(concept)
        if not concept_info:
            logger.warning(f"⚠️  概念未找到: {concept}")
            return pd.DataFrame()

        # 判断是否为SOFA相关概念
        is_sofa_concept = any(sofa_comp in concept.lower()
                            for sofa_comp in ['sofa', 'resp', 'coag', 'liver', 'cardio', 'cns', 'renal'])

        # 加载数据
        data_frames = []
        for db_name, sources in concept_info.sources.items():
            # 只处理当前数据库的源
            if db_name != self.database:
                continue

            for i, source in enumerate(sources):
                # 超详细的调试信息
                logger.debug(f"🔍 处理源{i}: 类型={type(source)}, 内容={repr(source)}")

                # 增强的类型检查和错误处理
                try:
                    # 检查sources列表本身的内容
                    if isinstance(source, str):
                        logger.error(f"❌ 源{i}是字符串: '{source}', 这表明存在序列化问题")
                        logger.error(f"❌ sources列表类型: {type(sources)}")
                        logger.error(f"❌ sources列表内容: {[type(s) for s in sources]}")

                        # 尝试重建概念源
                        try:
                            logger.info(f"🔄 尝试重建概念字典...")
                            # 重新获取概念信息
                            fresh_concept_info = self.concept_resolver.dictionary.get(concept)
                            if fresh_concept_info and 'miiv' in fresh_concept_info.sources:
                                fresh_sources = fresh_concept_info.sources['miiv']
                                logger.info(f"🔄 重建后的源: {fresh_sources}")
                                if i < len(fresh_sources):
                                    source = fresh_sources[i]
                                    logger.info(f"🔄 使用重建的源{i}: {source}")
                        except Exception as rebuild_error:
                            logger.error(f"❌ 重建失败: {rebuild_error}")

                        continue

                    if isinstance(source, dict):
                        table_name = source.get('table')
                        itemids = source.get('ids')
                        logger.debug(f"✅ 源{i}是字典: table={table_name}, itemids={itemids}")
                    elif hasattr(source, 'table'):
                        table_name = source.table
                        itemids = source.ids if hasattr(source, 'ids') else None
                        logger.debug(f"✅ 源{i}是对象: table={table_name}, itemids={itemids}")
                    else:
                        # 处理意外的数据类型
                        logger.error(f"❌ 概念源{i}类型错误: {type(source)}")
                        logger.error(f"❌ 源{i}内容: {repr(source)}")
                        logger.error(f"❌ 源{i}属性: {[attr for attr in dir(source) if not attr.startswith('_')]}")
                        continue

                    # 验证提取的值
                    if not table_name:
                        logger.warning(f"⚠️  无法获取table_name，跳过源{i}: {source}")
                        continue

                except Exception as e:
                    logger.error(f"❌ 处理概念源{i}时出错: {e}")
                    logger.error(f"❌ 错误类型: {type(e)}")
                    logger.error(f"❌ 源{i}类型: {type(source)}")
                    logger.error(f"❌ 源{i}内容: {repr(source)}")

                    # 打印完整的堆栈跟踪用于调试
                    import traceback
                    logger.error(f"❌ 堆栈跟踪: {traceback.format_exc()}")
                    continue

                # 创建过滤器
                filters = []
                if patient_ids:
                    # 获取正确的ID列名
                    id_col = self._get_id_column_for_table(table_name)
                    # 直接使用FilterOp，避免导入问题
                    from pyricu.datasource import FilterSpec, FilterOp
                    filters.append(FilterSpec(id_col, FilterOp.IN, patient_ids))

                # 使用优化的数据源加载
                try:
                    if hasattr(self.datasource, 'load_table_optimized'):
                        df = self.datasource.load_table_optimized(
                            table_name=table_name,
                            columns=None,  # 让优化器决定需要的列
                            filters=filters,
                            concept_name=concept if is_sofa_concept else None,
                            verbose=False
                        )
                    else:
                        # 回退到原始方法
                        df = self.datasource.load_table(
                            table_name=table_name,
                            filters=filters,
                            verbose=False
                        )

                    if not df.empty:
                        # 应用itemid过滤（如果没有在加载阶段应用）
                        if itemids and 'itemid' in df.columns:
                            original_count = len(df)
                            df = df[df['itemid'].isin(itemids)]
                            filtered_count = len(df)

                            if filtered_count == 0 and original_count > 0:
                                logger.warning(f"⚠️  itemid过滤后无数据: {table_name}, 期望itemids: {itemids}, 实际itemids: {sorted(df['itemid'].unique())}")
                            elif filtered_count < original_count:
                                logger.info(f"✅ itemid过滤: {table_name}, {original_count} → {filtered_count} 行")

                        data_frames.append(df)
                    else:
                        if itemids:
                            logger.warning(f"⚠️  表 {table_name} 为空，无法应用itemid过滤，期望itemids: {itemids}")

                except Exception as e:
                    logger.warning(f"⚠️  加载表失败 {table_name}: {e}")

        if not data_frames:
            return pd.DataFrame()

        # 合并数据帧
        if len(data_frames) == 1:
            combined_data = data_frames[0]
        else:
            combined_data = pd.concat(data_frames, ignore_index=True)

        # 应用时间窗口和聚合
        if interval or win_length or aggregate:
            combined_data = self._apply_time_processing(
                combined_data, interval, win_length, aggregate
            )

        return combined_data

    def _get_id_column_for_table(self, table_name: str) -> str:
        """获取表的ID列名"""
        id_mapping = {
            'chartevents': 'stay_id',
            'labevents': 'stay_id',
            'inputevents': 'stay_id',
            'outputevents': 'stay_id',
            'procedureevents': 'stay_id',
            'microbiologyevents': 'stay_id',
            'icustays': 'stay_id',
            'patients': 'subject_id'
        }
        return id_mapping.get(table_name, 'stay_id')

    def _apply_time_processing(
        self,
        df: pd.DataFrame,
        interval: Optional[Union[str, pd.Timedelta]],
        win_length: Optional[Union[str, pd.Timedelta]],
        aggregate: Optional[Union[str, Dict]]
    ) -> pd.DataFrame:
        """应用时间处理和聚合"""
        # 这里应该实现时间处理逻辑
        # 为简化，暂时返回原始数据
        return df

    def benchmark_concepts(
        self,
        concepts: List[str],
        patient_ids: List[int],
        **kwargs
    ) -> Dict[str, Any]:
        """
        对概念加载进行基准测试

        Args:
            concepts: 概念列表
            patient_ids: 患者ID列表
            **kwargs: 其他参数

        Returns:
            基准测试结果
        """
        logger.info(f"🏁 开始概念加载基准测试")

        # 测试优化版本
        self.benchmark_results = []
        optimized_start = time.time()

        optimized_result = self.load_concepts_optimized(
            concepts=concepts,
            patient_ids=patient_ids,
            benchmark=True,
            **kwargs
        )

        optimized_time = time.time() - optimized_start

        # 汇总结果
        benchmark_summary = {
            'concepts': concepts,
            'patient_count': len(patient_ids),
            'total_time': optimized_time,
            'concept_details': self.benchmark_results,
            'total_rows': sum(detail['rows'] for detail in self.benchmark_results)
        }

        logger.info(f"📊 基准测试完成:")
        logger.info(f"   概念数: {len(concepts)}")
        logger.info(f"   患者数: {len(patient_ids)}")
        logger.info(f"   总时间: {optimized_time:.2f}秒")
        logger.info(f"   总行数: {benchmark_summary['total_rows']:,}")

        return benchmark_summary


# 全局优化加载器实例
_global_optimized_loader = None


def get_optimized_loader(
    data_path: Optional[Union[str, Path]] = None,
    database: Optional[str] = None,
    **kwargs
) -> OptimizedConceptLoader:
    """
    获取全局优化加载器实例

    Args:
        data_path: 数据路径
        database: 数据库类型
        **kwargs: 其他参数

    Returns:
        优化加载器实例
    """
    global _global_optimized_loader

    if _global_optimized_loader is None:
        _global_optimized_loader = OptimizedConceptLoader(
            data_path=data_path,
            database=database,
            **kwargs
        )

    return _global_optimized_loader


def load_concepts_optimized(
    concepts: Union[str, List[str]],
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Optional[Union[str, pd.Timedelta]] = None,
    win_length: Optional[Union[str, pd.Timedelta]] = None,
    aggregate: Optional[Union[str, Dict]] = None,
    keep_components: bool = False,
    use_sofa2: bool = False,
    merge: bool = True,
    verbose: bool = False,
    benchmark: bool = False,
    **kwargs
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    优化的概念加载函数（便捷接口）

    Args:
        concepts: 概念名称或列表
        patient_ids: 患者ID
        database: 数据库类型
        data_path: 数据路径
        interval: 时间间隔
        win_length: 窗口长度
        aggregate: 聚合方式
        keep_components: 保留组件
        use_sofa2: 使用SOFA2
        merge: 是否合并结果
        verbose: 详细日志
        benchmark: 是否进行基准测试
        **kwargs: 其他参数

    Returns:
        加载的概念数据
    """
    loader = get_optimized_loader(data_path, database)
    return loader.load_concepts_optimized(
        concepts=concepts,
        patient_ids=patient_ids,
        interval=interval,
        win_length=win_length,
        aggregate=aggregate,
        keep_components=keep_components,
        use_sofa2=use_sofa2,
        merge=merge,
        verbose=verbose,
        benchmark=benchmark,
        **kwargs
    )


# SOFA专用的便捷函数
def load_sofa_optimized(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    keep_components: bool = False,
    use_sofa2: bool = False,
    verbose: bool = False,
    **kwargs
) -> pd.DataFrame:
    """
    优化的SOFA评分加载

    Args:
        patient_ids: 患者ID
        database: 数据库类型
        data_path: 数据路径
        keep_components: 保留组件
        use_sofa2: 使用SOFA2
        verbose: 详细日志
        **kwargs: 其他参数

    Returns:
        SOFA评分数据
    """
    concept = 'sofa2' if use_sofa2 else 'sofa'
    return load_concepts_optimized(
        concepts=concept,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        keep_components=keep_components,
        use_sofa2=use_sofa2,
        verbose=verbose,
        **kwargs
    )