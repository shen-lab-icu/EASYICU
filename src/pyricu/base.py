"""
Base classes and utilities for pyricu

This module provides unified base classes that consolidate common functionality
from across the codebase, reducing code duplication.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
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
            # Load concept dictionaries
            if dict_path is None:
                # Use built-in dictionaries
                dicts = [load_dictionary(include_sofa2=use_sofa2)]

                
            elif isinstance(dict_path, (list, tuple)):
                dicts = [load_dictionary(p) for p in dict_path]
            else:
                dicts = [load_dictionary(dict_path)]

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

    def load_concepts(
        self,
        concepts: Union[str, List[str]],
        patient_ids: Optional[List] = None,
        interval: Optional[Union[str, pd.Timedelta]] = None,
        win_length: Optional[Union[str, pd.Timedelta]] = None,
        aggregate: Optional[Union[str, Dict]] = None,
        keep_components: bool = False,
        merge: bool = True,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load concept data using the unified interface.

        This method consolidates the loading logic from multiple API implementations.
        """
        try:
            # Convert concepts to list
            if isinstance(concepts, str):
                concepts = [concepts]

            # Convert time parameters
            if isinstance(interval, str):
                interval = pd.Timedelta(interval)
            if isinstance(win_length, str):
                win_length = pd.Timedelta(win_length)

            # Load concepts efficiently - load all concepts at once
            if self.verbose:
                logger.info(f"Loading {len(concepts)} concepts: {', '.join(concepts)}")

            # Load all concepts in one call
            all_results = self.concept_resolver.load_concepts(
                concepts,
                self.datasource,
                patient_ids=patient_ids,
                interval=interval,
                win_length=win_length,
                aggregate=aggregate,
                keep_components=keep_components,  # 🔧 FIX: 传递 keep_components 参数
                **kwargs
            )

            # Handle different return formats
            if isinstance(all_results, dict):
                # Multiple concepts returned separately
                results = all_results
                if not merge:
                    return results
                else:
                    if self.verbose:
                        logger.info("Merging concept results")
                    return self._merge_concepts(results, keep_components)
            else:
                # Already merged result
                return all_results

        except Exception as e:
            raise RuntimeError(f"Failed to load concepts {concepts}: {e}")

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


def get_default_data_path(database: str) -> Optional[Path]:
    """Get default data path for database (convenience function)"""
    loader = BaseICULoader(database=database, verbose=False)
    return loader.data_path


def detect_database_type() -> str:
    """Auto-detect database type from environment (convenience function)"""
    loader = BaseICULoader(verbose=False)
    return loader.database