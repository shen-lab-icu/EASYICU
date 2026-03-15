"""
Base classes and utilities for pyricu

This module provides unified base classes that consolidate common functionality
from across the codebase, reducing code duplication.
"""

from __future__ import annotations

import os
import logging
import threading
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Sequence
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd

from .datasource import ICUDataSource
from .concept import ConceptResolver, ConceptDictionary
from .resources import load_data_sources, load_dictionary
from .cache_manager import get_cache_manager
from .parallel_config import get_global_config, get_runtime_load_strategy
from .table import ICUTable

logger = logging.getLogger(__name__)


def _batch_patient_count(batch_ids: Union[List, Dict, None]) -> int:
    """Best-effort patient count for a chunk payload."""
    if isinstance(batch_ids, dict):
        for value in batch_ids.values():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return len(value)
        return 0
    if isinstance(batch_ids, Sequence) and not isinstance(batch_ids, (str, bytes)):
        return len(batch_ids)
    return 0

class BaseICULoader:
    """
    Unified base loader class that consolidates common initialization and loading logic.

    This class replaces the multiple initialization patterns found in:
    - quickstart.py (ICUQuickLoader)
    - api.py (load_concepts function)
    - api_enhanced.py (cached loading)
    - api_unified.py (UnifiedConceptLoader)
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        database: Optional[str] = None,
        dict_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        use_sofa2: bool = False,
        verbose: bool = False,
    ):
        """Initialize the unified loader

        Args:
            data_path: Path to ICU data (auto-detected if None)
            database: Database type ('miiv', 'mimic', 'eicu', 'hirid', 'aumc')
            dict_path: Custom concept dictionary path(s)
            use_sofa2: Whether to load SOFA2 dictionary
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        # 检测数据库类型 - 如果用户没有指定，先尝试从路径推断
        self.database = self._detect_database(database, data_path)
        self.data_path = self._setup_data_path(data_path, self.database)
        self._dict_path = dict_path
        self._use_sofa2 = use_sofa2

        # Check and prepare data (convert CSV to Parquet if needed)
        self._ensure_data_ready()

        # Initialize data source
        self._init_datasource()

        # Initialize concept system
        self._init_concept_system(dict_path, use_sofa2)

        # Register caches with global cache manager
        self._register_caches()

        # Thread-local storage for per-worker concept resolvers
        self._thread_local_resolver = threading.local()

    def _detect_database(self, database: Optional[str], data_path: Optional[Union[str, Path]] = None) -> str:
        """Detect database type from data_path, environment or use default
        
        优先级:
        1. 用户显式指定的 database 参数
        2. 从 data_path 路径推断（检查路径名和数据文件）
        3. 环境变量
        4. 默认值 miiv
        """
        if database:
            return database
        
        # 尝试从 data_path 推断数据库类型
        if data_path:
            path = Path(data_path)
            path_str = str(path).lower()
            
            # 1. 检查路径名称中是否包含数据库标识
            db_patterns = {
                'eicu': ['eicu', 'eicu-crd'],
                'aumc': ['aumc', 'amsterdam'],
                'hirid': ['hirid'],
                'miiv': ['miiv', 'mimiciv', 'mimic-iv', 'mimic_iv'],
                'mimic': ['mimic', 'mimic-iii', 'mimic_iii', 'mimiciii'],
            }
            for db_name, patterns in db_patterns.items():
                if any(p in path_str for p in patterns):
                    if self.verbose:
                        logger.info(f"Auto-detected database: {db_name} from path: {path}")
                    return db_name
            
            # 2. 检查数据文件来推断数据库类型
            if path.is_dir():
                marker_files = {
                    'eicu': ['patient.parquet', 'patient.csv', 'patient.csv.gz', 'vitalPeriodic.parquet'],
                    'aumc': ['numericitems', 'admissions.parquet'],
                    'miiv': ['chartevents', 'icustays.parquet'],
                    'hirid': ['general_table.csv', 'general_table.parquet', 'observations'],  # 🔧 FIX: 正确的表名
                }
                for db_name, markers in marker_files.items():
                    if any((path / m).exists() for m in markers):
                        # 额外确认：避免误判
                        if db_name == 'eicu' and (path / 'patient.parquet').exists():
                            # 确认是 eicu 而不是其他有 patient 表的数据库
                            if not (path / 'chartevents').exists():
                                if self.verbose:
                                    logger.info(f"Auto-detected database: {db_name} from data files in: {path}")
                                return db_name
                        elif db_name != 'eicu':
                            if self.verbose:
                                logger.info(f"Auto-detected database: {db_name} from data files in: {path}")
                            return db_name

        # Check environment variables
        for db_name in ['miiv', 'mimic', 'eicu', 'hirid', 'aumc']:
            env_var = f'{db_name.upper()}_PATH'
            if os.getenv(env_var):
                if self.verbose:
                    logger.info(f"Auto-detected database: {db_name} from {env_var}")
                return db_name

        # Default
        if self.verbose:
            logger.info("Using default database: miiv")
        return 'miiv'

    def _setup_data_path(self, data_path: Optional[Union[str, Path]], database: str) -> Path:
        """Setup and validate data path
        
        智能处理数据路径：
        - 如果用户传入完整的数据库路径（包含数据文件），直接使用
        - 如果用户传入的是基础路径（如 /home/1_publicData/icu_databases），自动查找数据库子目录
        """
        if data_path:
            user_path = Path(data_path)
            
            # 检查用户路径是否直接包含数据文件（如 admissions.parquet, numericitems/ 等）
            if user_path.is_dir():
                # 检查是否是有效的数据库目录（包含特征文件）
                # AUMC 特征文件: admissions.csv/parquet, numericitems/
                # MIIV 特征文件: admissions.csv/parquet, chartevents/
                # eICU 特征文件: patient.csv, vitalPeriodic.csv
                # MIMIC-III 特征文件: icustays.parquet, chartevents_bucket/
                # SICdb 特征文件: cases.parquet, data_float_h_bucket/
                marker_files = {
                    'aumc': ['admissions.csv', 'admissions.parquet', 'numericitems'],
                    'miiv': ['admissions.csv', 'admissions.parquet', 'chartevents'],
                    'eicu': ['patient.csv', 'patient.csv.gz', 'vitalPeriodic.csv'],
                    'hirid': ['general.csv', 'observations'],
                    'mimic': ['icustays.parquet', 'chartevents_bucket', 'labevents_bucket'],  # MIMIC-III
                    'mimic_demo': ['icustays.parquet', 'chartevents'],  # MIMIC-III demo
                    'sic': ['cases.parquet', 'data_float_h_bucket', 'laboratory_bucket'],  # SICdb
                }
                
                db_markers = marker_files.get(database, [])
                is_valid_db_dir = any((user_path / marker).exists() for marker in db_markers)
                
                if is_valid_db_dir:
                    if self.verbose:
                        logger.info(f"Using user-provided database path: {user_path}")
                    return user_path
                
                # 如果不是有效的数据库目录，尝试查找子目录
                # AUMC 特殊处理：通常在 aumc/1.0.2/ 子目录
                # MIIV 特殊处理：通常在 mimiciv/3.1/ 子目录
                # eICU 特殊处理：通常在 eicu/2.0.1/ 子目录
                # HiRID 特殊处理：通常在 hirid/1.1.1/ 子目录
                # MIMIC-III 特殊处理：通常在 mimiciii/1.4/ 子目录
                # SICdb 特殊处理：通常在 sicdb/1.0.6/ 子目录
                # 先尝试精确版本匹配，再尝试通用目录
                possible_subpaths = [
                    user_path / database,  # /base/aumc
                    user_path / database / '1.0.2',  # /base/aumc/1.0.2 (AUMC)
                    user_path / database / '3.1',    # /base/miiv/3.1 (MIIV)
                    user_path / database / '2.0.1',  # /base/eicu/2.0.1 (eICU)
                    user_path / database / '2.0',    # /base/eicu/2.0 (eICU old)
                    user_path / database / '1.1.1',  # /base/hirid/1.1.1 (HiRID)
                    # 支持 mimiciv 命名变体
                    user_path / 'mimiciv' / '3.1',
                    # MIMIC-III 支持
                    user_path / 'mimiciii' / '1.4',
                    user_path / 'mimic' / '1.4',
                    # SICdb 支持
                    user_path / 'sicdb' / '1.0.6',
                    user_path / 'sic' / '1.0.6',
                ]
                
                # 如果没找到，尝试动态搜索子目录
                for subpath in possible_subpaths:
                    if subpath.is_dir():
                        is_valid = any((subpath / marker).exists() for marker in db_markers)
                        if is_valid:
                            if self.verbose:
                                logger.info(f"Auto-detected database path: {subpath} (from base: {user_path})")
                            return subpath
                
                # 回退：返回用户路径（可能导致后续错误，但保持向后兼容）
                if self.verbose:
                    logger.warning(f"Could not find valid {database} data in {user_path}, using as-is")
                return user_path
            
            return user_path

        # Check environment variables
        # 1. 首先检查数据库专用的环境变量（如 MIMIC_PATH）
        env_var = f'{database.upper()}_PATH'
        path = os.getenv(env_var)
        if path:
            if self.verbose:
                logger.info(f"Using path from {env_var}: {path}")
            return Path(path)
        
        # 2. 检查 RICU_DATA_PATH 通用环境变量（需要与数据库目录映射）
        ricu_data_path = os.getenv('RICU_DATA_PATH')
        if ricu_data_path:
            base_path = Path(ricu_data_path)
            # 数据库名称到目录名的映射
            db_dir_mapping = {
                'mimic': ['mimiciii/1.4', 'mimic/1.4', 'mimiciii'],
                'mimic_demo': ['mimic_demo', 'mimiciii_demo'],
                'miiv': ['mimiciv/3.1', 'miiv/3.1', 'mimiciv'],
                'eicu': ['eicu/2.0.1', 'eicu/2.0', 'eicu'],
                'eicu_demo': ['eicu_demo'],
                'aumc': ['aumc/1.0.2', 'aumc'],
                'hirid': ['hirid/1.1.1', 'hirid'],
                'sic': ['sicdb/1.0.6', 'sic/1.0.6', 'sicdb', 'sic'],
            }
            
            for subdir in db_dir_mapping.get(database, [database]):
                candidate = base_path / subdir
                if candidate.is_dir():
                    if self.verbose:
                        logger.info(f"Using path from RICU_DATA_PATH: {candidate}")
                    return candidate

        # Check production data paths from project_config
        try:
            from .project_config import get_data_path
            prod_path = get_data_path(source='production', database=database)
            if prod_path and prod_path.exists():
                if self.verbose:
                    logger.info(f"Using production data path: {prod_path}")
                return prod_path
        except Exception as e:
            if self.verbose:
                logger.debug(f"Could not get production path from project_config: {e}")

        # Check common paths
        common_paths = [
            Path.home() / 'data' / database,
            Path('/data') / database,
            Path('.') / 'data' / database,
        ]

        for path in common_paths:
            if path.exists():
                if self.verbose:
                    logger.info(f"Found existing path: {path}")
                return path

        # Return default path (may not exist)
        default_path = Path('./data') / database
        if self.verbose:
            logger.info(f"Using default path: {default_path}")
        return default_path

    def _ensure_data_ready(self):
        """Ensure data files are ready (convert CSV to Parquet if needed)
        
        This method checks if Parquet files exist for the database's tables.
        If only CSV/CSV.GZ files exist, a warning will be logged.
        Use DataConverter or CLI to convert files before loading.
        """
        try:
            from .data_converter import DataConverter
            
            converter = DataConverter(
                data_path=self.data_path,
                database=self.database,
                verbose=False  # Suppress verbose output for status check
            )
            
            # Check status without auto-converting
            is_ready, missing = converter.is_ready()
            
            if not is_ready:
                # Log warning about missing parquet files
                logger.warning(
                    f"⚠️ {len(missing)} CSV files need to be converted to Parquet for optimal performance.\n"
                    f"   Run: python -m pyricu.data_converter {self.data_path}\n"
                    f"   Or use: DataConverter('{self.data_path}').ensure_parquet_ready()"
                )
                if self.verbose:
                    for msg in missing[:5]:
                        logger.warning(f"   - {msg}")
                    if len(missing) > 5:
                        logger.warning(f"   ... and {len(missing) - 5} more")
            elif self.verbose:
                logger.info(f"✅ All data files are ready in {self.data_path}")
            
        except ImportError:
            if self.verbose:
                logger.debug("data_converter module not available, skipping data preparation check")
        except Exception as e:
            # Don't fail initialization if data check fails
            if self.verbose:
                logger.warning(f"Data preparation check failed: {e}")

    def _init_datasource(self):
        """Initialize the data source"""
        try:
            registry = load_data_sources()
            config = registry.get(self.database)
            if not config:
                raise ValueError(f"Unknown database: {self.database}")

            self.datasource = ICUDataSource(config=config, base_path=self.data_path)

            if self.verbose:
                logger.info(f"Initialized datasource for {self.database}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize datasource: {e}")

    def _init_concept_system(self, dict_path: Optional[Union[str, Path, List[Union[str, Path]]]], use_sofa2: bool):
        """Initialize the concept system"""
        try:
            if dict_path is None:
                env_override = os.getenv("PYRICU_DICT_PATH") or os.getenv("PYRICU_DICT_DIR")
                if env_override:
                    dict_path = env_override

            dicts: List[ConceptDictionary] = [load_dictionary(include_sofa2=use_sofa2)]

            if dict_path is not None:
                if isinstance(dict_path, (list, tuple)):
                    sources = list(dict_path)
                else:
                    sources = [dict_path]

                for source in sources:
                    dicts.append(self._load_dict_source(source))

            # Create merged dictionary
            if len(dicts) == 1:
                self.concept_dict = dicts[0]
            else:
                # Merge multiple dictionaries
                merged = dicts[0].copy()
                for dict_obj in dicts[1:]:
                    merged.update(dict_obj)
                self.concept_dict = merged

            # Initialize resolver
            self.concept_resolver = ConceptResolver(
                dictionary=self.concept_dict
            )

            if self.verbose:
                concept_count = len(list(self.concept_dict.keys()))
                logger.info(f"Initialized concept system with {concept_count} concepts")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize concept system: {e}")

    def _register_caches(self):
        """Register caches with global cache manager"""
        try:
            cache_manager = get_cache_manager()

            # Register data source cache
            if hasattr(self, 'datasource') and self.datasource:
                cache_manager.register_memory_cache(self.datasource)

            # Register concept resolver cache
            if hasattr(self, 'concept_resolver') and self.concept_resolver:
                cache_manager.register_memory_cache(self.concept_resolver)

            if self.verbose:
                logger.info("✅ 已注册缓存到全局缓存管理器")

        except Exception as e:
            if self.verbose:
                logger.warning(f"⚠️  缓存注册失败: {e}")
            # 不影响主要功能，继续运行

    def clear_cache(self):
        """Clear all caches to free memory
        
        This is useful when processing data in batches to ensure each batch
        uses fresh data and doesn't accumulate memory from previous batches.
        """
        if hasattr(self, 'concept_resolver') and self.concept_resolver:
            self.concept_resolver.clear_table_cache(keep_concept_cache=False)
        
        # Also try to clear datasource cache if it has one
        if hasattr(self, 'datasource') and self.datasource:
            if hasattr(self.datasource, 'clear_cache'):
                self.datasource.clear_cache()
            elif hasattr(self.datasource, '_table_cache'):
                self.datasource._table_cache.clear()
            elif hasattr(self.datasource, '_cache'):
                self.datasource._cache.clear()

    def _create_resolver_clone(self) -> ConceptResolver:
        """Create a fresh ConceptResolver sharing the same dictionary."""
        return ConceptResolver(dictionary=self.concept_dict)

    def _create_loader_clone(self) -> "BaseICULoader":
        """Create a lightweight fresh loader for isolated chunk execution."""
        loader = BaseICULoader.__new__(BaseICULoader)
        loader.verbose = self.verbose
        loader.database = self.database
        loader.data_path = self.data_path
        loader._dict_path = getattr(self, '_dict_path', None)
        loader._use_sofa2 = getattr(self, '_use_sofa2', False)
        loader.datasource = ICUDataSource(
            config=self.datasource.config,
            base_path=self.data_path,
        )
        loader.concept_dict = self.concept_dict
        loader.concept_resolver = ConceptResolver(dictionary=self.concept_dict)
        loader._thread_local_resolver = threading.local()
        return loader

    def _get_thread_resolver(self) -> ConceptResolver:
        """Lazily create per-thread concept resolvers for parallel batches."""
        if not hasattr(self, '_thread_local_resolver'):
            self._thread_local_resolver = threading.local()
        resolver = getattr(self._thread_local_resolver, 'resolver', None)
        if resolver is None:
            resolver = self._create_resolver_clone()
            self._thread_local_resolver.resolver = resolver
        return resolver

    def _limit_blas_threads(self) -> Dict[str, Optional[str]]:
        """Force single-threaded BLAS during Python-level threading."""
        env_vars = [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ]
        original: Dict[str, Optional[str]] = {}
        for var in env_vars:
            original[var] = os.environ.get(var)
            os.environ[var] = "1"
        return original

    def _restore_blas_threads(self, state: Dict[str, Optional[str]]) -> None:
        for var, value in state.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

    def _resolve_parallel_workers(self, requested: Optional[int]) -> int:
        """Determine how many patient-chunk workers to spawn by default.
        
        ⚡ 性能优化: 由于Python GIL和锁竞争，多线程反而会降低性能
        默认使用单线程，除非明确指定
        """
        if isinstance(requested, int) and requested > 0:
            return requested

        env_value = os.getenv("PYRICU_PARALLEL_WORKERS")
        if env_value:
            try:
                env_workers = int(env_value)
                if env_workers > 0:
                    return env_workers
            except ValueError:
                logger.warning("Invalid PYRICU_PARALLEL_WORKERS=%s, ignoring", env_value)

        # ⚡ 默认单线程以避免GIL竞争和锁开销
        # 用户可通过环境变量PYRICU_PARALLEL_WORKERS或参数显式启用并行
        default_workers = 1
        
        return default_workers

    def _resolve_parallel_backend(self, backend: Optional[str]) -> str:
        """Select execution backend for patient chunk parallelism."""
        if backend:
            normalized = backend.strip().lower()
            if normalized in {"thread", "process"}:
                return normalized
            if normalized != "auto":
                logger.warning("Unknown parallel_backend '%s', falling back to auto", backend)

        env_backend = os.getenv("PYRICU_PARALLEL_BACKEND")
        if env_backend:
            normalized = env_backend.strip().lower()
            if normalized in {"thread", "process"}:
                return normalized

        if os.name == "nt":
            return "thread"
        return "process"

    def _load_dict_source(self, source: Union[str, Path, ConceptDictionary]) -> ConceptDictionary:
        """Load a dictionary from a custom source."""
        if isinstance(source, ConceptDictionary):
            return source

        if isinstance(source, (str, Path)):
            path = Path(str(source))
            if path.exists():
                if path.is_dir():
                    return load_dictionary(directories=[path])
                if path.is_file():
                    return ConceptDictionary.from_json(path)

        # Fallback to treating the string as a packaged resource name
        return load_dictionary(str(source))

    def load_concepts(
        self,
        concepts: Union[str, List[str]],
        patient_ids: Optional[List] = None,
        interval: Optional[Union[str, pd.Timedelta]] = '1h',  # ricu默认: hours(1L)
        win_length: Optional[Union[str, pd.Timedelta]] = None,
        aggregate: Optional[Union[str, Dict]] = None,
        keep_components: bool = False,
        merge: bool = True,
        ricu_compatible: bool = True,  # 默认启用ricu.R兼容模式
        chunk_size: Optional[int] = None,
        progress: bool = False,
        parallel_workers: Optional[int] = None,
        concept_workers: Optional[int] = None,  # 改为Optional，支持自动检测
        parallel_backend: str = "auto",
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load concept data using the unified interface.

        This method consolidates the loading logic from multiple API implementations.
        """
        try:
            kwargs = dict(kwargs)
            if isinstance(concepts, str):
                concepts = [concepts]

            if isinstance(interval, str):
                interval = pd.Timedelta(interval)
            if isinstance(win_length, str):
                win_length = pd.Timedelta(win_length)

            num_patients = _batch_patient_count(patient_ids)
            runtime_strategy = get_runtime_load_strategy(
                concepts,
                num_patients=num_patients if num_patients > 0 else None,
                chunk_size=chunk_size,
                requested_concept_workers=concept_workers,
                requested_parallel_workers=parallel_workers,
                requested_backend=parallel_backend if parallel_backend != "auto" else None,
                config=get_global_config(),
            )
            effective_concept_workers = int(runtime_strategy["concept_workers"])
            
            if self.verbose:
                logger.info(f"Loading {len(concepts)} concepts: {', '.join(concepts)}")
                if effective_concept_workers > 1:
                    logger.info(f"⚡ Auto-optimized: concept_workers={effective_concept_workers}")

            batches = self._build_patient_batches(patient_ids, chunk_size)
            if batches:
                worker_count = (
                    self._resolve_parallel_workers(parallel_workers)
                    if parallel_workers is not None
                    else int(runtime_strategy["parallel_workers"])
                )
                return self._load_concepts_chunked(
                    concepts,
                    batches,
                    interval,
                    win_length,
                    aggregate,
                    keep_components,
                    merge,
                    ricu_compatible,
                    progress,
                    worker_count,
                    effective_concept_workers,
                    str(runtime_strategy["parallel_backend"]),
                    kwargs,
                )

            return self._load_concepts_once(
                concepts,
                patient_ids,
                interval,
                win_length,
                aggregate,
                keep_components,
                merge,
                ricu_compatible,
                effective_concept_workers,
                kwargs,
                preserve_cache=bool(runtime_strategy["preserve_concept_cache"]),
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load concepts {concepts}: {e}")
        # 🚀 优化：移除finally中的强制清除缓存
        # _load_concepts_once 已经有条件地管理缓存，无需在此再清除
        # 这样批量加载多个概念时可以共享表缓存，大幅提升性能

    def _merge_concepts(self, results: Dict[str, pd.DataFrame], keep_components: bool) -> pd.DataFrame:
        """Merge multiple concept DataFrames"""
        if not results:
            return pd.DataFrame()

        merged_df = None
        id_cols = None

        for concept, df in results.items():
            if df.empty:
                continue

            if id_cols is None:
                id_cols = [
                    col for col in df.columns
                    if col in ['stay_id', 'subject_id', 'patientunitstayid', 'admissionid', 'patientid', 'icustay_id', 'CaseID']
                ]
                if not id_cols:
                    id_cols = [df.columns[0]]
                merged_df = df
            else:
                merge_keys = self._get_merge_keys(merged_df, df, id_cols)
                merged_df = self._outer_merge_frames(merged_df, df, merge_keys, concept)

        return merged_df if merged_df is not None else pd.DataFrame()

    @staticmethod
    def _get_merge_keys(left: pd.DataFrame, right: pd.DataFrame, id_cols: List[str]) -> List[str]:
        """Choose stable merge keys, preferring shared ID + time columns."""
        shared_ids = [col for col in id_cols if col in left.columns and col in right.columns]
        shared_time = [
            col for col in left.columns
            if col in right.columns and ('time' in col.lower() or col in {'date', 'day'})
        ]
        return shared_ids + [col for col in shared_time if col not in shared_ids]

    @staticmethod
    def _has_unique_keys(frame: pd.DataFrame, merge_keys: List[str]) -> bool:
        if not merge_keys:
            return False
        if any(col not in frame.columns for col in merge_keys):
            return False
        return not frame.duplicated(subset=merge_keys).any()

    def _outer_merge_frames(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
        merge_keys: List[str],
        concept_name: str,
    ) -> pd.DataFrame:
        """Merge two concept frames with a low-copy fast path for unique keys."""
        if not merge_keys:
            return pd.concat([left, right], ignore_index=True, sort=False, copy=False)

        overlapping_cols = [
            col for col in left.columns
            if col in right.columns and col not in merge_keys
        ]
        can_fast_join = (
            not overlapping_cols
            and self._has_unique_keys(left, merge_keys)
            and self._has_unique_keys(right, merge_keys)
        )

        if can_fast_join:
            left_indexed = left.set_index(merge_keys, drop=True)
            right_indexed = right.set_index(merge_keys, drop=True)
            return (
                pd.concat(
                    [left_indexed, right_indexed],
                    axis=1,
                    join='outer',
                    sort=False,
                    copy=False,
                )
                .reset_index()
            )

        return pd.merge(
            left,
            right,
            on=merge_keys,
            how='outer',
            suffixes=('', f'_{concept_name}')
        )

    def _load_concepts_once(
        self,
        concepts: List[str],
        patient_ids: Optional[Union[List, Dict]],
        interval: Optional[pd.Timedelta],
        win_length: Optional[pd.Timedelta],
        aggregate: Optional[Union[str, Dict]],
        keep_components: bool,
        merge: bool,
        ricu_compatible: bool,
        concept_workers: int,
        extra_kwargs: Dict[str, Any],
        preserve_cache: bool = False,
        resolver: Optional[ConceptResolver] = None,
        use_thread_resolver: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        params = dict(extra_kwargs)
        verbose_flag = params.pop("verbose", self.verbose)
        
        resolver_obj: ConceptResolver
        if resolver is not None:
            resolver_obj = resolver
        elif use_thread_resolver:
            resolver_obj = self._get_thread_resolver()
        else:
            resolver_obj = self.concept_resolver

        try:
            result = resolver_obj.load_concepts(
                concepts,
                self.datasource,
                merge=merge,
                patient_ids=patient_ids,
                interval=interval,
                win_length=win_length,
                aggregate=aggregate,
                keep_components=keep_components,
                ricu_compatible=ricu_compatible,
                concept_workers=concept_workers,
                verbose=verbose_flag,
                **params,
            )
            # Compatibility fix: when running in ricu_compatible mode, R ricu
            # applies sed_impute='max' for total GCS (tgcs) when ett_gcs == TRUE.
            # Some tgcs are computed via sum_components and therefore miss the
            # sed_impute adjustment. Apply the adjustment here on the merged
            # DataFrame so tgcs matches R ricu semantics when both columns
            # are present.
            try:
                if ricu_compatible and isinstance(result, pd.DataFrame):
                    if 'tgcs' in result.columns and 'ett_gcs' in result.columns:
                        mask = result['ett_gcs'].where(result['ett_gcs'].notna(), False).astype(bool)
                        if mask.any():
                            result.loc[mask, 'tgcs'] = 15.0
            except Exception:
                # Do not fail loading if this adjustment fails
                pass
        finally:
            # 🚀 优化：只清除表缓存，保留概念数据缓存以加速批量加载
            # 表缓存可能很大（原始数据），但概念缓存较小（聚合后的数据）
            # 这允许在连续的 load_concepts 调用之间共享概念缓存（如 sofa 和 sofa2 共享 fio2, plt 等）
            resolver_obj.clear_table_cache(keep_concept_cache=preserve_cache)

        if isinstance(result, dict):
            if not merge:
                return result
            if self.verbose:
                logger.info("Merging concept results")
            return self._merge_concepts(result, keep_components)
        return result

    def _load_concepts_chunked(
        self,
        concepts: List[str],
        batches: List[Union[List, Dict]],
        interval: Optional[pd.Timedelta],
        win_length: Optional[pd.Timedelta],
        aggregate: Optional[Union[str, Dict]],
        keep_components: bool,
        merge: bool,
        ricu_compatible: bool,
        progress: bool,
        parallel_workers: int,
        concept_workers: int,
        parallel_backend: str,
        extra_kwargs: Dict[str, Any],
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        aggregated_frames: List[pd.DataFrame] = []
        aggregated_dict: Dict[str, List[pd.DataFrame]] = defaultdict(list)
        aggregated_meta: Dict[str, Dict[str, Any]] = {}
        total_batches = len(batches)
        if parallel_workers is None or parallel_workers <= 0:
            parallel_workers = 1
        if total_batches:
            parallel_workers = max(1, min(parallel_workers, total_batches))
        backend = self._resolve_parallel_backend(parallel_backend)
        
        # 🚀 优化：在处理前预加载大表（避免重复I/O）
        if backend == "thread" and parallel_workers > 1 and total_batches > 1:
            logger.info(
                f"🚀 启用多线程优化({backend}): {parallel_workers}线程处理{total_batches}批次"
            )

            # ❌ 临时禁用预加载：预加载逻辑有bug，会在load_table时无限递归
            # TODO: 修复预加载逻辑后重新启用
            total_patients = sum(_batch_patient_count(batch) for batch in batches)
            logger.info(f"⚡ 数据规模({total_patients}患者)，预加载功能暂时禁用")
            # if len(all_patient_ids) >= 1000:
            #     preload_tables = ['chartevents', 'labevents', 'outputevents', 'procedureevents']
            #     logger.
            #     self.datasource.preload_tables(preload_tables, patient_ids=all_patient_ids)
            # else:
            #     logger.info(f"⚡ 小规模数据({len(all_patient_ids)}患者)，跳过预加载以提升性能")
        elif backend == "process" and parallel_workers > 1 and total_batches > 1:
            logger.info(
                f"🚀 启用多进程优化: {parallel_workers}进程处理{total_batches}批次"
            )

            # ❌ 临时禁用预加载
            total_patients = sum(_batch_patient_count(batch) for batch in batches)
            logger.info(f"⚡ 多进程模式({total_patients}患者)，预加载功能暂时禁用")
            # preload_tables = ['chartevents', 'labevents', 'outputevents', 'procedureevents']
            # logger.info(f"📦 多进程模式预加载大表: {', '.join(preload_tables)}")
            # self.datasource.preload_tables(preload_tables, patient_ids=all_patient_ids)

        def _capture_meta(table: ICUTable) -> Dict[str, Any]:
            return {
                "id_columns": list(table.id_columns),
                "index_column": table.index_column,
                "value_column": table.value_column,
                "unit_column": table.unit_column,
                "time_columns": list(table.time_columns),
            }

        def _accumulate(chunk_result):
            if isinstance(chunk_result, dict):
                for name, frame in chunk_result.items():
                    if frame is None:
                        continue
                    meta = None
                    if isinstance(frame, ICUTable):
                        meta = _capture_meta(frame)
                        frame_data = frame.data
                    else:
                        frame_data = frame
                    if frame_data is not None and not getattr(frame_data, "empty", False):
                        aggregated_dict[name].append(frame_data)
                        if meta and name not in aggregated_meta:
                            aggregated_meta[name] = meta
            else:
                if chunk_result is not None and not getattr(chunk_result, "empty", False):
                    aggregated_frames.append(chunk_result)

        def _load_chunk_isolated(batch_ids):
            batch_loader = self._create_loader_clone()
            try:
                return batch_loader._load_concepts_once(
                    concepts,
                    batch_ids,
                    interval,
                    win_length,
                    aggregate,
                    keep_components,
                    merge,
                    ricu_compatible,
                    concept_workers,
                    extra_kwargs,
                    preserve_cache=False,
                    resolver=batch_loader.concept_resolver,
                )
            finally:
                batch_loader.clear_cache()

        if parallel_workers and parallel_workers > 1:
            blas_state: Optional[Dict[str, Optional[str]]] = None
            if backend == "thread":
                blas_state = self._limit_blas_threads()
            try:
                if backend == "process":
                    worker_payload = {
                        "database": self.database,
                        "data_path": str(self.data_path) if self.data_path is not None else None,
                        "dict_path": self._dict_path,
                        "use_sofa2": getattr(self, "_use_sofa2", False),
                        "datasource_config": self.datasource.config,
                        "concept_dict": self.concept_dict,
                        "verbose": self.verbose,
                    }
                    with ProcessPoolExecutor(
                        max_workers=parallel_workers,
                        initializer=_init_parallel_chunk_worker,
                        initargs=(worker_payload,)
                    ) as executor:
                        future_map = {
                            executor.submit(
                                _process_chunk_task,
                                (
                                    concepts,
                                    batch_ids,
                                    interval,
                                    win_length,
                                    aggregate,
                                    keep_components,
                                    merge,
                                    ricu_compatible,
                                    concept_workers,
                                    extra_kwargs,
                                ),
                            ): idx
                            for idx, batch_ids in enumerate(batches, start=1)
                        }
                        for future in as_completed(future_map):
                            idx = future_map[future]
                            chunk_result = future.result()
                            _accumulate(chunk_result)
                            if progress:
                                pct = (idx / total_batches) * 100.0
                                logger.info(
                                    "Chunked load %s: %d/%d (%.1f%%)",
                                    ", ".join(concepts),
                                    idx,
                                    total_batches,
                                    pct,
                                )
                else:
                    executor_params = dict(max_workers=parallel_workers)
                    with ThreadPoolExecutor(**executor_params) as executor:
                        future_map = {
                            executor.submit(
                                _load_chunk_isolated,
                                batch_ids,
                            ): idx
                            for idx, batch_ids in enumerate(batches, start=1)
                        }
                        for future in as_completed(future_map):
                            idx = future_map[future]
                            chunk_result = future.result()
                            _accumulate(chunk_result)
                            if progress:
                                pct = (idx / total_batches) * 100.0
                                logger.info(
                                    "Chunked load %s: %d/%d (%.1f%%)",
                                    ", ".join(concepts),
                                    idx,
                                    total_batches,
                                    pct,
                                )
            finally:
                if blas_state is not None:
                    self._restore_blas_threads(blas_state)
        else:
            for idx, batch_ids in enumerate(batches, start=1):
                chunk_result = _load_chunk_isolated(batch_ids)
                _accumulate(chunk_result)
                if progress:
                    pct = (idx / total_batches) * 100.0
                    logger.info(
                        "Chunked load %s: %d/%d (%.1f%%)",
                        ", ".join(concepts),
                        idx,
                        total_batches,
                        pct,
                    )

        self.concept_resolver.clear_table_cache()

        if aggregated_dict:
            combined: Dict[str, Any] = {}
            for name, frames in aggregated_dict.items():
                combined_frame = (
                    pd.concat(frames, ignore_index=True, sort=False, copy=False)
                    if len(frames) > 1
                    else frames[0]
                )
                if not merge and name in aggregated_meta:
                    meta = aggregated_meta[name]
                    combined[name] = ICUTable(
                        data=combined_frame,
                        id_columns=meta.get("id_columns") or [],
                        index_column=meta.get("index_column"),
                        value_column=meta.get("value_column"),
                        unit_column=meta.get("unit_column"),
                        time_columns=meta.get("time_columns") or [],
                    )
                else:
                    combined[name] = combined_frame
            if merge:
                return self._merge_concepts(combined, keep_components)
            return combined

        if aggregated_frames:
            return (
                pd.concat(aggregated_frames, ignore_index=True, sort=False, copy=False)
                if len(aggregated_frames) > 1
                else aggregated_frames[0]
            )

        return pd.DataFrame()

    def _build_patient_batches(
        self,
        patient_ids: Optional[Union[List, Dict]],
        chunk_size: Optional[int],
    ) -> Optional[List[Union[List, Dict]]]:
        if not chunk_size or chunk_size <= 0 or patient_ids is None:
            return None

        if isinstance(patient_ids, dict):
            if len(patient_ids) != 1:
                return None
            key, values = next(iter(patient_ids.items()))
            seq = self._normalize_patient_ids(values)
            if seq is None or len(seq) <= chunk_size:
                return None
            return [
                {key: seq[i : i + chunk_size]}
                for i in range(0, len(seq), chunk_size)
            ]

        if isinstance(patient_ids, Sequence) and not isinstance(patient_ids, (str, bytes)):
            seq = self._normalize_patient_ids(patient_ids)
            if seq is None or len(seq) <= chunk_size:
                return None
            return [seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]

        return None

    @staticmethod
    def _normalize_patient_ids(values: Union[Sequence, pd.Series]) -> Optional[List]:
        if values is None:
            return None
        try:
            seq = list(dict.fromkeys(values))
        except TypeError:
            seq = list(values)
        return seq

def get_default_data_path(database: str) -> Optional[Path]:
    """Get default data path for database (convenience function)"""
    loader = BaseICULoader(database=database, verbose=False)
    return loader.data_path

def detect_database_type() -> str:
    """Auto-detect database type from environment (convenience function)"""
    loader = BaseICULoader(verbose=False)
    return loader.database

_PROCESS_WORKER_LOADER: Optional[BaseICULoader] = None

def _init_parallel_chunk_worker(payload: Dict[str, Any]) -> None:
    """Initializer for process-based chunk workers."""
    global _PROCESS_WORKER_LOADER
    loader = BaseICULoader.__new__(BaseICULoader)
    loader.verbose = payload.get("verbose", False)
    loader.database = payload.get("database")
    data_path = payload.get("data_path")
    loader.data_path = Path(data_path) if data_path else None
    loader._dict_path = payload.get("dict_path")
    loader._use_sofa2 = payload.get("use_sofa2", False)
    loader.datasource = ICUDataSource(
        config=payload["datasource_config"],
        base_path=loader.data_path,
    )
    loader.concept_dict = payload["concept_dict"]
    loader.concept_resolver = ConceptResolver(dictionary=loader.concept_dict)
    loader._thread_local_resolver = threading.local()
    _PROCESS_WORKER_LOADER = loader

def _process_chunk_task(args: tuple) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Execute a patient chunk inside a worker process."""
    if _PROCESS_WORKER_LOADER is None:
        raise RuntimeError("Chunk worker not initialized")

    (
        concepts,
        batch_ids,
        interval,
        win_length,
        aggregate,
        keep_components,
        merge,
        ricu_compatible,
        concept_workers,
        extra_kwargs,
    ) = args

    batch_loader = _PROCESS_WORKER_LOADER._create_loader_clone()
    try:
        return batch_loader._load_concepts_once(
            concepts,
            batch_ids,
            interval,
            win_length,
            aggregate,
            keep_components,
            merge,
            ricu_compatible,
            concept_workers,
            extra_kwargs,
            preserve_cache=False,
            resolver=batch_loader.concept_resolver,
        )
    finally:
        batch_loader.clear_cache()
