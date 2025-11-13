"""
Unified API for pyricu - consolidating multiple implementations.

This module provides a unified interface for pyricu operations,
consolidating the multiple implementations found across the codebase.
"""

from typing import Dict, List, Optional, Union, Any
import warnings
import logging
from pathlib import Path
import pandas as pd

try:
    from .concept import ConceptResolver, ConceptDictionary
except ImportError:
    ConceptResolver = None
    ConceptDictionary = None

try:
    from .datasource import ICUDataSource
except ImportError:
    ICUDataSource = None

try:
    from .resources import load_data_sources, load_dictionary
except ImportError:
    load_data_sources = None
    load_dictionary = None

try:
    from .common_utils import DataFrameUtils, ValidationUtils
except ImportError:
    DataFrameUtils = None
    ValidationUtils = None

try:
    from .memory_optimizer import MemoryMonitor, get_memory_monitor
except ImportError:
    MemoryMonitor = None
    get_memory_monitor = None

logger = logging.getLogger(__name__)


class UnifiedConceptLoader:
    """
    Unified concept loading implementation that consolidates multiple approaches.

    This class replaces the multiple load_concepts implementations found in:
    - concept.py
    - quickstart.py
    - load_concepts.py
    - api.py
    - callbacks.py
    """

    def __init__(self,
                 data_source: Optional[ICUDataSource] = None,
                 dictionary: Optional[ConceptDictionary] = None,
                 memory_optimized: bool = False,
                 monitor: Optional[MemoryMonitor] = None):
        """
        Initialize unified concept loader.

        Args:
            data_source: Data source instance
            dictionary: Concept dictionary
            memory_optimized: Enable memory optimizations
            monitor: Memory monitor instance
        """
        self.data_source = data_source
        self.dictionary = dictionary or load_dictionary()
        self.memory_optimized = memory_optimized
        self.monitor = monitor or get_memory_monitor()

        # Initialize resolver if data source is provided
        self.resolver = None
        if data_source:
            self.resolver = ConceptResolver(data_source, self.dictionary)

    def load_concepts(self,
                     concepts: Union[str, List[str]],
                     *,
                     database: Optional[str] = None,
                     patient_ids: Optional[List[Any]] = None,
                     merge: bool = True,
                     time_window: Optional[int] = None,
                     use_chunking: bool = None,
                     max_memory_mb: Optional[int] = None,
                     **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Unified concept loading method.

        This method consolidates functionality from:
        - concept.py:load_concepts()
        - api.py:load_concepts()
        - quickstart.py:load_concepts()
        - load_concepts.py:load_concepts()

        Args:
            concepts: Concept name(s) to load
            database: Database name ('miiv', 'eicu', etc.)
            patient_ids: Optional patient ID filter
            merge: Whether to merge concepts into single DataFrame
            time_window: Time window in hours
            use_chunking: Force enable/disable chunking (auto-detected if None)
            max_memory_mb: Memory limit for loading
            **kwargs: Additional parameters passed to resolver

        Returns:
            DataFrame(s) with concept data

        Raises:
            ValueError: If concepts cannot be loaded
            MemoryError: If memory limit is exceeded
        """
        if self.monitor:
            self.monitor.log_memory_usage("load_concepts start")

        # Prepare concept list
        if isinstance(concepts, str):
            concept_list = [concepts]
        else:
            concept_list = list(concepts)

        # Set up data source if needed
        if self.resolver is None:
            if database is None:
                raise ValueError("Database must be specified when no data source is provided")

            data_sources = load_data_sources()
            if database not in data_sources:
                raise ValueError(f"Unknown database: {database}")

            from .config import DataSourceConfig
            config = data_sources.get(database)
            self.data_source = ICUDataSource(config)
            self.resolver = ConceptResolver(self.dictionary)

        # Determine chunking strategy
        if use_chunking is None:
            # Auto-detect based on data size and memory pressure
            memory_pressure = self.monitor.check_memory_pressure()
            use_chunking = (
                self.memory_optimized or
                memory_pressure in ['HIGH', 'EMERGENCY'] or
                len(patient_ids or []) > 1000
            )

        try:
            if use_chunking:
                return self._load_concepts_chunked(
                    concept_list,
                    patient_ids=patient_ids,
                    merge=merge,
                    time_window=time_window,
                    max_memory_mb=max_memory_mb,
                    **kwargs
                )
            else:
                return self._load_concepts_standard(
                    concept_list,
                    patient_ids=patient_ids,
                    merge=merge,
                    time_window=time_window,
                    max_memory_mb=max_memory_mb,
                    **kwargs
                )

        except MemoryError as e:
            if not use_chunking:
                logger.warning("Memory error, retrying with chunking")
                return self._load_concepts_chunked(
                    concept_list,
                    patient_ids=patient_ids,
                    merge=merge,
                    time_window=time_window,
                    max_memory_mb=max_memory_mb,
                    **kwargs
                )
            else:
                raise e

        finally:
            if self.monitor:
                self.monitor.log_memory_usage("load_concepts complete")
                self.monitor.cleanup_if_needed()

    def _load_concepts_standard(self,
                               concept_list: List[str],
                               patient_ids: Optional[List[Any]] = None,
                               merge: bool = True,
                               time_window: Optional[int] = None,
                               max_memory_mb: Optional[int] = None,
                               **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Standard concept loading (consolidated from multiple implementations)."""
        results = {}

        # Load each concept
        for concept in concept_list:
            try:
                if self.monitor:
                    self.monitor.log_memory_usage(f"Load {concept}")

                concept_data = self.resolver.load_concepts(
                    [concept],
                    self.data_source,
                    patient_ids=patient_ids,
                    time_window=time_window,
                    **kwargs
                )

                if not concept_data.empty:
                    # Optimize memory usage
                    if self.memory_optimized:
                        concept_data = DataFrameUtils.optimize_dtypes(concept_data)

                    results[concept] = concept_data

                # Check memory usage
                if max_memory_mb:
                    current_memory = sum(
                        df.memory_usage(deep=True).sum()
                        for df in results.values()
                    ) / (1024 * 1024)

                    if current_memory > max_memory_mb:
                        raise MemoryError(f"Memory limit exceeded: {current_memory:.1f}MB > {max_memory_mb}MB")

            except Exception as e:
                logger.error(f"Failed to load concept {concept}: {e}")
                continue

        if not results:
            return pd.DataFrame() if merge else {}

        # Merge if requested
        if merge and len(results) > 1:
            return self._merge_concepts(results)
        elif merge and len(results) == 1:
            return list(results.values())[0]
        else:
            return results

    def _load_concepts_chunked(self,
                              concept_list: List[str],
                              patient_ids: Optional[List[Any]] = None,
                              merge: bool = True,
                              time_window: Optional[int] = None,
                              max_memory_mb: Optional[int] = None,
                              **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Chunked concept loading for memory efficiency."""

        if not patient_ids:
            # If no patient filter, load concept data first to get patient IDs
            sample_data = self.resolver.load_concepts([concept_list[0]], self.data_source, **kwargs)
            if sample_data.empty:
                return pd.DataFrame() if merge else {}
            patient_ids = sample_data.iloc[:, 0].unique().tolist()

        # Determine chunk size
        chunk_size = self.monitor.config.chunk_size
        if max_memory_mb:
            # Adjust chunk size based on memory limit
            estimated_per_patient = 50  # KB per patient (rough estimate)
            chunk_size = min(chunk_size, max_memory_mb * 1024 // estimated_per_patient)

        logger.info(f"Loading {len(concept_list)} concepts for {len(patient_ids)} patients in chunks of {chunk_size}")

        # Process in chunks
        all_results = {}
        for i in range(0, len(patient_ids), chunk_size):
            chunk_patients = patient_ids[i:i + chunk_size]

            if self.monitor:
                self.monitor.log_memory_usage(f"Processing chunk {i//chunk_size + 1}")

            # Load concepts for this chunk
            chunk_results = self._load_concepts_standard(
                concept_list,
                patient_ids=chunk_patients,
                merge=False,  # Don't merge within chunks
                time_window=time_window,
                max_memory_mb=max_memory_mb,
                **kwargs
            )

            # Combine with previous results
            for concept, data in chunk_results.items():
                if concept not in all_results:
                    all_results[concept] = []
                all_results[concept].append(data)

            # Cleanup
            del chunk_results
            self.monitor.cleanup_if_needed()

        # Combine chunk results
        final_results = {}
        for concept, data_list in all_results.items():
            if data_list:
                final_results[concept] = pd.concat(data_list, ignore_index=True, sort=False)
                # Optimize final result
                if self.memory_optimized:
                    final_results[concept] = DataFrameUtils.optimize_dtypes(final_results[concept])

        if not final_results:
            return pd.DataFrame() if merge else {}

        # Merge if requested
        if merge:
            return self._merge_concepts(final_results)
        else:
            return final_results

    def _merge_concepts(self, concept_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple concept DataFrames.

        This consolidates merging logic from multiple implementations.
        """
        if not concept_data:
            return pd.DataFrame()

        # Sort concepts by name for consistent merge order
        sorted_concepts = sorted(concept_data.keys())

        # Start with the first concept
        result = concept_data[sorted_concepts[0]].copy()

        # Merge remaining concepts
        for concept in sorted_concepts[1:]:
            df = concept_data[concept]

            # Determine merge columns
            id_cols = ['stay_id']
            if 'time' in result.columns and 'time' in df.columns:
                merge_cols = id_cols + ['time']
                how = 'outer'
            else:
                merge_cols = id_cols
                how = 'left'

            # Use memory-efficient merge
            result = DataFrameUtils.memory_efficient_merge(
                result, df, on=merge_cols, how=how
            )

        return result

    def get_available_concepts(self) -> List[str]:
        """Get list of available concepts."""
        return list(self.dictionary.concepts.keys())

    def get_concept_info(self, concept: str) -> Dict[str, Any]:
        """Get information about a specific concept."""
        if concept not in self.dictionary.concepts:
            raise ValueError(f"Unknown concept: {concept}")

        concept_def = self.dictionary.concepts[concept]
        return {
            'name': concept,
            'description': concept_def.get('description', ''),
            'category': concept_def.get('category', ''),
            'sources': list(concept_def.get('sources', {}).keys()),
            'omop_id': concept_def.get('omopid'),
        }


# Global loader instance for convenience
_global_loader = None

def get_loader(memory_optimized: bool = False) -> UnifiedConceptLoader:
    """Get or create global concept loader."""
    global _global_loader
    if _global_loader is None:
        _global_loader = UnifiedConceptLoader(memory_optimized=memory_optimized)
    return _global_loader


# Backward compatible functions
def load_concepts(concepts: Union[str, List[str]],
                 database: Optional[str] = None,
                 **kwargs) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Backward compatible load_concepts function.

    This consolidates all previous load_concepts implementations.
    """
    loader = get_loader()
    return loader.load_concepts(concepts, database=database, **kwargs)


def load_concept(concept: str,
                database: Optional[str] = None,
                **kwargs) -> pd.DataFrame:
    """Load a single concept."""
    result = load_concepts(concept, database=database, merge=False, **kwargs)
    if isinstance(result, dict):
        return result.get(concept, pd.DataFrame())
    return result


def list_available_concepts(database: Optional[str] = None) -> List[str]:
    """List available concepts."""
    loader = get_loader()
    concepts = loader.get_available_concepts()

    if database is not None:
        # Filter concepts available for specific database
        filtered_concepts = []
        for concept in concepts:
            info = loader.get_concept_info(concept)
            if database in info['sources']:
                filtered_concepts.append(concept)
        return filtered_concepts

    return concepts


def get_concept_info(concept: str) -> Dict[str, Any]:
    """Get information about a concept."""
    loader = get_loader()
    return loader.get_concept_info(concept)


# Easy API functions (consolidating easy.py and api.py)
def load_sofa(database: str = 'miiv',
             patient_ids: Optional[List[Any]] = None,
             memory_optimized: bool = None,
             **kwargs) -> pd.DataFrame:
    """Load SOFA scores with unified API."""
    if memory_optimized is None:
        # Auto-detect based on system
        monitor = get_memory_monitor()
        memory_optimized = monitor.check_memory_pressure() != 'SAFE'

    loader = get_loader(memory_optimized=memory_optimized)
    return loader.load_concepts('sofa', database=database, patient_ids=patient_ids, **kwargs)


def load_sofa2(database: str = 'miiv',
              patient_ids: Optional[List[Any]] = None,
              memory_optimized: bool = None,
              **kwargs) -> pd.DataFrame:
    """Load SOFA-2 scores with unified API."""
    if memory_optimized is None:
        monitor = get_memory_monitor()
        memory_optimized = monitor.check_memory_pressure() != 'SAFE'

    loader = get_loader(memory_optimized=memory_optimized)
    return loader.load_concepts('sofa2', database=database, patient_ids=patient_ids, **kwargs)


def load_vitals(database: str = 'miiv',
               patient_ids: Optional[List[Any]] = None,
               memory_optimized: bool = None,
               **kwargs) -> pd.DataFrame:
    """Load vital signs with unified API."""
    vital_concepts = ['hr', 'sbp', 'dbp', 'temp', 'resp', 'spo2']

    if memory_optimized is None:
        monitor = get_memory_monitor()
        memory_optimized = monitor.check_memory_pressure() != 'SAFE'

    loader = get_loader(memory_optimized=memory_optimized)
    return loader.load_concepts(vital_concepts, database=database, patient_ids=patient_ids, **kwargs)


def load_labs(database: str = 'miiv',
             patient_ids: Optional[List[Any]] = None,
             memory_optimized: bool = None,
             **kwargs) -> pd.DataFrame:
    """Load lab values with unified API."""
    lab_concepts = ['wbc', 'hgb', 'plt', 'crea', 'bili', 'lact']

    if memory_optimized is None:
        monitor = get_memory_monitor()
        memory_optimized = monitor.check_memory_pressure() != 'SAFE'

    loader = get_loader(memory_optimized=memory_optimized)
    return loader.load_concepts(lab_concepts, database=database, patient_ids=patient_ids, **kwargs)