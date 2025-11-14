"""
Base classes and utilities for pyricu

This module provides unified base classes that consolidate common functionality
from across the codebase, reducing code duplication.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd

from .datasource import ICUDataSource
from .concept import ConceptResolver, ConceptDictionary
from .resources import load_data_sources, load_dictionary

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
        verbose: bool = False
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
        parallel_workers: int = 1,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load concept data using the unified interface.

        This method consolidates the loading logic from multiple API implementations.
        """
        try:
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
                    parallel_workers,
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
                kwargs,
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
        extra_kwargs: Dict[str, Any],
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        try:
            result = self.concept_resolver.load_concepts(
                concepts,
                self.datasource,
                patient_ids=patient_ids,
                interval=interval,
                win_length=win_length,
                aggregate=aggregate,
                keep_components=keep_components,
                ricu_compatible=ricu_compatible,
                **extra_kwargs,
            )
        finally:
            self.concept_resolver.clear_table_cache()

        if isinstance(result, dict):
            if not merge:
                return result
            if self.verbose:
                logger.info("Merging concept results")
            return self._merge_concepts(result, keep_components)
        return result

    def _load_concepts_once_worker(
        self,
        patient_ids: Optional[Union[List, Dict]],
        concepts: List[str],
        interval: Optional[pd.Timedelta],
        win_length: Optional[pd.Timedelta],
        aggregate: Optional[Union[str, Dict]],
        keep_components: bool,
        merge: bool,
        ricu_compatible: bool,
        extra_kwargs: Dict[str, Any],
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        worker = BaseICULoader(
            data_path=self.data_path,
            database=self.database,
            dict_path=self._dict_path,
            use_sofa2=self._use_sofa2,
            verbose=False,
        )
        return worker._load_concepts_once(
            concepts,
            patient_ids,
            interval,
            win_length,
            aggregate,
            keep_components,
            merge,
            ricu_compatible,
            extra_kwargs,
        )

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
        extra_kwargs: Dict[str, Any],
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        aggregated_frames: List[pd.DataFrame] = []
        aggregated_dict: Dict[str, List[pd.DataFrame]] = defaultdict(list)
        total_batches = len(batches)

        def _accumulate(chunk_result):
            if isinstance(chunk_result, dict):
                for name, frame in chunk_result.items():
                    if frame is not None and not frame.empty:
                        aggregated_dict[name].append(frame)
            else:
                if chunk_result is not None and not getattr(chunk_result, "empty", False):
                    aggregated_frames.append(chunk_result)

        if parallel_workers and parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                future_map = {
                    executor.submit(
                        self._load_concepts_once_worker,
                        batch_ids,
                        concepts,
                        interval,
                        win_length,
                        aggregate,
                        keep_components,
                        merge,
                        ricu_compatible,
                        extra_kwargs,
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
                    extra_kwargs,
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

        if aggregated_dict:
            combined = {
                name: pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
                for name, frames in aggregated_dict.items()
            }
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
