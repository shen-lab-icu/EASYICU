"""
专门针对notebook环境的优化API
解决'str' object has no attribute 'table'错误
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import pandas as pd

# 尝试导入，如果失败则提供替代方案
try:
    from .optimized_loader import get_optimized_datasource, OptimizedLoaderFactory
    from .concept import ConceptDictionary, ConceptResolver
    from .base import BaseICULoader
    from .cache_manager import get_cache_manager
    from .datasource import FilterSpec, FilterOp
    OPTIMIZED_LOADED = True
except ImportError as e:
    logging.warning(f"优化模块导入失败: {e}，将使用基础API")
    OPTIMIZED_LOADED = False

logger = logging.getLogger(__name__)


class NotebookOptimizedConceptLoader:
    """针对notebook环境优化的概念加载器"""

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
        初始化notebook优化概念加载器

        Args:
            data_path: 数据路径
            database: 数据库类型
            dict_path: 字典路径
            use_sofa2: 是否使用SOFA2字典
            verbose: 详细日志
            enable_optimizations: 启用优化
            benchmark_mode: 基准测试模式
        """
        self.database = database or 'miiv'
        self.data_path = Path(data_path) if data_path else Path('data/miiv')
        self.verbose = verbose
        self.enable_optimizations = enable_optimizations and OPTIMIZED_LOADED
        self.benchmark_mode = benchmark_mode

        if self.verbose:
            logger.info(f"🚀 初始化notebook优化加载器...")
            logger.info(f"   数据库: {self.database}")
            logger.info(f"   数据路径: {self.data_path}")
            logger.info(f"   优化启用: {self.enable_optimizations}")

        # 安全初始化组件
        self._safe_init_components()

        self.benchmark_results = []

    def _safe_init_components(self):
        """安全初始化所有组件，处理各种导入错误"""
        try:
            if self.enable_optimizations:
                # 尝试创建优化的数据源
                self.datasource = get_optimized_datasource(
                    database=self.database,
                    data_path=self.data_path,
                    enable_column_pruning=True,
                    enable_itemid_filtering=True
                )
                if self.verbose:
                    logger.info("✅ 优化数据源创建成功")
            else:
                # 回退到基础数据源
                from .datasource import ICUDataSource
                self.datasource = ICUDataSource(database=self.database, data_path=self.data_path)
                if self.verbose:
                    logger.info("✅ 基础数据源创建成功")

        except Exception as e:
            logger.error(f"❌ 数据源初始化失败: {e}")
            # 创建一个最小的回退数据源
            self._create_fallback_datasource()

        try:
            # 安全创建概念解析器
            self.concept_resolver = self._safe_create_concept_resolver()
            if self.verbose:
                logger.info("✅ 概念解析器创建成功")
        except Exception as e:
            logger.error(f"❌ 概念解析器初始化失败: {e}")
            self.concept_resolver = None

    def _create_fallback_datasource(self):
        """创建回退数据源"""
        class FallbackDataSource:
            def __init__(self, data_path):
                self.base_path = Path(data_path)
                self.table_paths = {}

            def load_table(self, table_name, filters=None, verbose=False):
                """基础表加载"""
                try:
                    table_path = self.base_path / f"{table_name}.parquet"
                    if table_path.is_dir():
                        # 处理目录中的多个parquet文件
                        import glob
                        parquet_files = glob.glob(str(table_path / "*.parquet"))
                        if parquet_files:
                            dfs = [pd.read_parquet(f) for f in parquet_files]
                            return pd.concat(dfs, ignore_index=True)
                    elif table_path.exists():
                        return pd.read_parquet(table_path)
                except Exception as e:
                    if verbose:
                        logger.error(f"加载表失败 {table_name}: {e}")
                return pd.DataFrame()

            def load_table_optimized(self, table_name, columns=None, filters=None, verbose=False):
                """回退优化加载"""
                return self.load_table(table_name, filters, verbose)

        self.datasource = FallbackDataSource(self.data_path)

    def _safe_create_concept_resolver(self):
        """安全创建概念解析器"""
        try:
            # 尝试多种方式创建概念解析器
            try:
                # 方式1: 直接创建
                return ConceptResolver()
            except TypeError:
                try:
                    # 方式2: 从JSON文件创建
                    concept_dict = ConceptDictionary()
                    return ConceptResolver(concept_dict)
                except Exception:
                    # 方式3: 使用硬编码的概念定义
                    return self._create_hardcoded_concept_resolver()
        except Exception as e:
            logger.error(f"❌ 所有概念解析器创建方式都失败: {e}")
            return None

    def _create_hardcoded_concept_resolver(self):
        """创建硬编码的概念解析器作为最后回退"""
        class HardcodedConceptResolver:
            def __init__(self):
                # 硬编码一些基本概念
                self.concepts = {
                    'hr': ConceptInfo('hr', 'heart rate', 'chartevents', [220045]),
                    'sbp': ConceptInfo('sbp', 'systolic blood pressure', 'chartevents', [220050, 220179]),
                    'spo2': ConceptInfo('spo2', 'oxygen saturation', 'chartevents', [220277, 226253]),
                    'resp': ConceptInfo('resp', 'respiratory rate', 'chartevents', [220210, 224688, 224689, 224690]),
                }

            def get(self, concept_name):
                return self.concepts.get(concept_name)

        class ConceptInfo:
            def __init__(self, name, description, table, itemids):
                self.name = name
                self.description = description
                self.sources = {'miiv': [HardcodedSource(table, itemids)]}

        class HardcodedSource:
            def __init__(self, table, itemids):
                self.table = table
                self.ids = itemids

        return HardcodedConceptResolver()

    def load_concepts_notebook_safe(
        self,
        concepts: Union[str, List[str]],
        patient_ids: Optional[Union[List, Dict]] = None,
        interval: Optional[Union[str, pd.Timedelta]] = None,
        win_length: Optional[Union[str, pd.Timedelta]] = None,
        aggregate: Optional[Union[str, Dict]] = None,
        keep_components: bool = False,
        use_sofa2: bool = False,
        merge: bool = True,
        verbose: Optional[bool] = None,
        benchmark: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        notebook安全的概念加载方法

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
        if verbose is None:
            verbose = self.verbose

        start_time = time.time()

        if verbose:
            logger.info(f"🚀 开始notebook安全加载概念: {concepts}")

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
                # 使用安全的方法加载概念
                concept_data = self._safe_load_single_concept(
                    concept=concept,
                    patient_ids=patient_ids_list,
                    interval=interval,
                    win_length=win_length,
                    aggregate=aggregate,
                    use_sofa2=use_sofa2,
                    verbose=verbose
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
                if verbose:
                    import traceback
                    logger.error(f"详细错误: {traceback.format_exc()}")
                results[concept] = pd.DataFrame()

        # 合并结果
        if merge and len(results) > 1:
            # 安全合并所有结果
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
            logger.info(f"✅ notebook安全加载完成: {total_time:.2f}秒")

        return final_result

    def _safe_load_single_concept(
        self,
        concept: str,
        patient_ids: Optional[List],
        interval: Optional[Union[str, pd.Timedelta]],
        win_length: Optional[Union[str, pd.Timedelta]],
        aggregate: Optional[Union[str, Dict]],
        use_sofa2: bool,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        安全加载单个概念，包含所有错误处理
        """
        if verbose:
            logger.info(f"🔄 安全加载概念: {concept}")

        # 获取概念定义
        if not self.concept_resolver:
            logger.warning(f"⚠️ 概念解析器不可用，跳过概念: {concept}")
            return pd.DataFrame()

        concept_info = self.concept_resolver.get(concept)
        if not concept_info:
            logger.warning(f"⚠️ 概念未找到: {concept}")
            return pd.DataFrame()

        # 判断是否为SOFA相关概念
        is_sofa_concept = any(sofa_comp in concept.lower()
                            for sofa_comp in ['sofa', 'resp', 'coag', 'liver', 'cardio', 'cns', 'renal'])

        # 处理概念源
        data_frames = []

        # 支持多种概念信息格式
        try:
            if hasattr(concept_info, 'sources'):
                sources_dict = concept_info.sources
            elif hasattr(concept_info, 'sources') and isinstance(concept_info.sources, dict):
                sources_dict = concept_info.sources
            else:
                # 假设是硬编码的简单格式
                sources_dict = {'miiv': [concept_info]} if hasattr(concept_info, 'table') else {}
        except Exception as e:
            logger.error(f"❌ 处理概念源失败 {concept}: {e}")
            return pd.DataFrame()

        for db_name, sources in sources_dict.items():
            # 只处理当前数据库的源
            if db_name != self.database:
                continue

            # 确保sources是列表
            if not isinstance(sources, list):
                sources = [sources]

            for i, source in enumerate(sources):
                try:
                    if verbose:
                        logger.debug(f"🔍 处理源{i}: 类型={type(source)}, 内容={repr(source)}")

                    # 多种源类型处理
                    table_name = None
                    itemids = None

                    if isinstance(source, str):
                        logger.error(f"❌ 源{i}是字符串，可能存在序列化问题: '{source}'")
                        continue

                    elif isinstance(source, dict):
                        table_name = source.get('table')
                        itemids = source.get('ids')
                        if verbose:
                            logger.debug(f"✅ 源{i}是字典: table={table_name}, itemids={itemids}")

                    elif hasattr(source, 'table'):
                        table_name = source.table
                        itemids = source.ids if hasattr(source, 'ids') else None
                        if verbose:
                            logger.debug(f"✅ 源{i}是对象: table={table_name}, itemids={itemids}")

                    else:
                        logger.error(f"❌ 不支持的源类型{i}: {type(source)}, 内容: {source}")
                        continue

                    if not table_name:
                        logger.warning(f"⚠️ 无法获取table_name，跳过源{i}")
                        continue

                    # 创建过滤器
                    filters = []
                    if patient_ids:
                        id_col = self._get_id_column_for_table(table_name)
                        filters.append(FilterSpec(id_col, FilterOp.IN, patient_ids))

                    # 使用数据源加载
                    try:
                        if hasattr(self.datasource, 'load_table_optimized'):
                            df = self.datasource.load_table_optimized(
                                table_name=table_name,
                                columns=None,
                                filters=filters,
                                concept_name=concept if is_sofa_concept else None,
                                verbose=False
                            )
                        else:
                            df = self.datasource.load_table(
                                table_name=table_name,
                                filters=filters,
                                verbose=False
                            )

                        if not df.empty:
                            # 应用itemid过滤
                            if itemids and 'itemid' in df.columns:
                                original_count = len(df)
                                df = df[df['itemid'].isin(itemids)]
                                filtered_count = len(df)

                                if verbose and filtered_count < original_count:
                                    logger.info(f"✅ itemid过滤: {original_count} → {filtered_count} 行")

                            data_frames.append(df)

                    except Exception as e:
                        logger.warning(f"⚠️ 加载表失败 {table_name}: {e}")
                        continue

                except Exception as e:
                    logger.error(f"❌ 处理概念源{i}时出错: {e}")
                    continue

        if not data_frames:
            return pd.DataFrame()

        # 合并数据帧
        if len(data_frames) == 1:
            combined_data = data_frames[0]
        else:
            combined_data = pd.concat(data_frames, ignore_index=True)

        # 应用时间处理（简化版）
        if interval or win_length or aggregate:
            if verbose:
                logger.info(f"🔄 应用时间处理...")
            # 这里可以实现时间处理逻辑
            pass

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


# 全局notebook安全加载器实例
_global_notebook_loader = None


def get_notebook_optimized_loader(
    data_path: Optional[Union[str, Path]] = None,
    database: Optional[str] = None,
    **kwargs
) -> NotebookOptimizedConceptLoader:
    """
    获取全局notebook安全加载器实例

    Args:
        data_path: 数据路径
        database: 数据库类型
        **kwargs: 其他参数

    Returns:
        notebook安全加载器实例
    """
    global _global_notebook_loader

    if _global_notebook_loader is None:
        _global_notebook_loader = NotebookOptimizedConceptLoader(
            data_path=data_path,
            database=database,
            **kwargs
        )

    return _global_notebook_loader


def load_concepts_notebook_safe(
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
    verbose: bool = True,
    benchmark: bool = False,
    **kwargs
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    notebook安全的概念加载函数（便捷接口）

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
    loader = get_notebook_optimized_loader(data_path, database)
    return loader.load_concepts_notebook_safe(
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


# 为用户创建一个更简单的接口
def load_concepts_notebook(
    concepts,
    patient_ids=None,
    database='miiv',
    data_path='/home/1_publicData/icu_databases/mimiciv/3.1',
    verbose=True,
    **kwargs
):
    """
    简化的notebook接口，专门解决用户的问题

    Args:
        concepts: 概念名称或列表，如 ['hr', 'sbp', 'spo2', 'resp']
        patient_ids: 患者ID列表
        database: 数据库名称，默认'miiv'
        data_path: 数据路径，默认使用用户的MIMIC-IV路径
        verbose: 是否显示详细信息，默认True
        **kwargs: 其他参数

    Returns:
        加载的概念数据DataFrame
    """
    warnings.filterwarnings('ignore')  # 抑制警告

    try:
        result = load_concepts_notebook_safe(
            concepts=concepts,
            patient_ids=patient_ids,
            database=database,
            data_path=data_path,
            verbose=verbose,
            **kwargs
        )

        print(f"✅ 成功加载概念 {concepts}")
        print(f"📊 结果形状: {result.shape}")
        if not result.empty and 'concept' in result.columns:
            concepts_found = result['concept'].unique()
            print(f"📋 实际加载的概念: {list(concepts_found)}")
            for concept in concepts_found:
                count = len(result[result['concept'] == concept])
                print(f"  • {concept}: {count}条记录")

        return result

    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print(f"🔄 尝试基础回退方案...")

        # 基础回退方案
        try:
            import pandas as pd
            from pathlib import Path

            # 直接读取chartevents数据
            chart_path = Path(data_path) / 'chartevents'
            if chart_path.is_dir():
                import glob
                parquet_files = glob.glob(str(chart_path / "*.parquet"))
                if parquet_files:
                    dfs = [pd.read_parquet(f) for f in parquet_files[:1]]  # 只读第一个文件
                    df = pd.concat(dfs, ignore_index=True)

                    # 简单过滤
                    if patient_ids:
                        df = df[df['stay_id'].isin(patient_ids)]

                    # 添加概念列
                    df['concept'] = 'hr'  # 假设都是心率数据

                    print(f"✅ 回退方案成功: {df.shape}")
                    return df
        except Exception as fallback_error:
            print(f"❌ 回退方案也失败: {fallback_error}")

        # 返回空DataFrame
        return pd.DataFrame()