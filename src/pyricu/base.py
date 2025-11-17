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
from .table import ICUTable

logger = logging.getLogger(__name__)


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
        self.database = self._detect_database(database)
        self.data_path = self._setup_data_path(data_path, self.database)
        self._dict_path = dict_path
        self._use_sofa2 = use_sofa2

        # Initialize data source
        self._init_datasource()

        # Initialize concept system
        self._init_concept_system(dict_path, use_sofa2)

        # Register caches with global cache manager
        self._register_caches()

        # Thread-local storage for per-worker concept resolvers
        self._thread_local_resolver = threading.local()

    def _detect_database(self, database: Optional[str]) -> str:
        """Detect database type from environment or use default"""
        if database:
            return database

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
        """Setup and validate data path"""
        if data_path:
            return Path(data_path)

        # Check environment variables
        env_var = f'{database.upper()}_PATH'
        path = os.getenv(env_var)
        if path:
            if self.verbose:
                logger.info(f"Using path from {env_var}: {path}")
            return Path(path)

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

    def _create_resolver_clone(self) -> ConceptResolver:
        """Create a fresh ConceptResolver sharing the same dictionary."""
        return ConceptResolver(dictionary=self.concept_dict)

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
        interval: Optional[Union[str, pd.Timedelta]] = None,
        win_length: Optional[Union[str, pd.Timedelta]] = None,
        aggregate: Optional[Union[str, Dict]] = None,
        keep_components: bool = False,
        merge: bool = True,
        ricu_compatible: bool = False,  # 新增：ricu.R兼容模式
        chunk_size: Optional[int] = None,
        progress: bool = False,
        parallel_workers: Optional[int] = None,
        concept_workers: int = 1,
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

            if self.verbose:
                logger.info(f"Loading {len(concepts)} concepts: {', '.join(concepts)}")

            batches = self._build_patient_batches(patient_ids, chunk_size)
            if batches:
                worker_count = self._resolve_parallel_workers(parallel_workers)
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
                    concept_workers,
                    parallel_backend,
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
                concept_workers,
                kwargs,
                preserve_cache=False,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load concepts {concepts}: {e}")
        finally:
            if hasattr(self, "concept_resolver"):
                self.concept_resolver.clear_table_cache()

    def _merge_concepts(self, results: Dict[str, pd.DataFrame], keep_components: bool) -> pd.DataFrame:
        """Merge multiple concept DataFrames"""
        if not results:
            return pd.DataFrame()

        # Start with first DataFrame
        merged_df = None
        id_cols = None

        for concept, df in results.items():
            if df.empty:
                continue

            # Identify ID columns from first non-empty DataFrame
            if id_cols is None:
                id_cols = [col for col in df.columns if col in ['stay_id', 'subject_id', 'patientunitstayid', 'admissionid', 'patientid']]
                if not id_cols:
                    id_cols = [df.columns[0]]  # Use first column as fallback
                merged_df = df.copy()
            else:
                # Merge on ID columns
                time_cols = [col for col in df.columns if 'time' in col.lower()]
                merge_on = list(id_cols)
                if time_cols:
                    merge_on.extend(time_cols)

                # Remove duplicate merge columns from df
                df_merge = df.drop(columns=[col for col in merge_on if col in df.columns and col != merge_on[0]])

                merged_df = pd.merge(
                    merged_df,
                    df_merge,
                    on=id_cols,
                    how='outer',
                    suffixes=('', f'_{concept}')
                )

        return merged_df if merged_df is not None else pd.DataFrame()

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
        
        # 🚀 优化：当加载多个相关概念时保留缓存（如SOFA的多个子概念）
        should_preserve_cache = preserve_cache or len(concepts) > 1
        
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
        finally:
            # 只有在不需要保留缓存时才清除
            if not should_preserve_cache:
                resolver_obj.clear_table_cache()

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
            
            # 智能预加载：只在患者数量足够时才预加载
            all_patient_ids = []
            for batch in batches:
                if isinstance(batch, dict):
                    all_patient_ids.extend(batch.get('stay_id', []))
                else:
                    all_patient_ids.extend(batch)

            # 只有患者数量足够多时才预加载，避免小数据集的性能开销
            if len(all_patient_ids) >= 1000:  # 提高阈值，减少不必要的预加载
                preload_tables = ['chartevents', 'labevents', 'outputevents', 'procedureevents']
                logger.info(f"📦 大规模数据({len(all_patient_ids)}患者)，预加载大表: {', '.join(preload_tables)}")
                self.datasource.preload_tables(preload_tables, patient_ids=all_patient_ids)
            else:
                logger.info(f"⚡ 小规模数据({len(all_patient_ids)}患者)，跳过预加载以提升性能")
        elif backend == "process" and parallel_workers > 1 and total_batches > 1:
            logger.info(
                f"🚀 启用多进程优化: {parallel_workers}进程处理{total_batches}批次"
            )

            # 为多进程模式也预加载大表，避免重复I/O
            all_patient_ids = []
            for batch in batches:
                if isinstance(batch, dict):
                    all_patient_ids.extend(batch.get('stay_id', []))
                else:
                    all_patient_ids.extend(batch)

            preload_tables = ['chartevents', 'labevents', 'outputevents', 'procedureevents']
            logger.info(f"📦 多进程模式预加载大表: {', '.join(preload_tables)}")
            self.datasource.preload_tables(preload_tables, patient_ids=all_patient_ids)

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
                                self._load_concepts_once,
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
                                True,
                                None,
                                True,
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
                chunk_result = self._load_concepts_once(
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
                    preserve_cache=True,
                    resolver=self.concept_resolver,
                )
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
                    pd.concat(frames, ignore_index=True)
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
                pd.concat(aggregated_frames, ignore_index=True)
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

    return _PROCESS_WORKER_LOADER._load_concepts_once(
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
        preserve_cache=True,
    )
