"""
Memory-optimized data source operations for 16GB systems.

This module provides memory-efficient alternatives to the standard data source operations,
specifically designed for systems with limited memory.
"""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator
import pandas as pd
import warnings

from .datasource import ICUDataSource, FilterSpec
from .memory_optimizer import MemoryMonitor, MemoryEfficientTable, get_memory_monitor


class OptimizedDataSource(ICUDataSource):
    """Memory-optimized ICU data source for 16GB systems."""

    def __init__(self, config, **kwargs):
        # Set memory-optimized defaults
        kwargs.setdefault('enable_cache', True)
        kwargs.setdefault('format_priority', ['parquet', 'csv'])  # Avoid FST for memory efficiency

        super().__init__(config, **kwargs)

        # Memory optimization components
        self.monitor = kwargs.get('monitor') or get_memory_monitor()
        self.table_ops = MemoryEfficientTable(self.monitor)

        # Override cache settings for 16GB systems
        self._cache_size_limit = self.monitor.config.cache_limit
        self._max_cached_tables = 10  # Limit number of cached tables

    def load_table_optimized(self,
                           table_name: str,
                           *,
                           columns: Optional[List[str]] = None,
                           patient_ids: Optional[List[Any]] = None,
                           chunk_size: Optional[int] = None,
                           optimize_dtypes: bool = True,
                           use_filtering: bool = True,
                           **kwargs) -> pd.DataFrame:
        """Load table with memory optimizations."""

        if self.monitor:
            self.monitor.log_memory_usage(f"Load {table_name}")

        # Check if we should use chunked loading
        if chunk_size and chunk_size < self.monitor.config.chunk_size:
            return self._load_table_chunked(
                table_name, columns=columns, patient_ids=patient_ids,
                chunk_size=chunk_size, optimize_dtypes=optimize_dtypes, **kwargs
            )

        # Apply memory pressure check
        pressure = self.monitor.check_memory_pressure()
        if pressure == 'EMERGENCY':
            warnings.warn("Emergency memory pressure - clearing all caches")
            self.clear_cache()
            gc.collect()

        # Check cache first
        cache_key = self._get_cache_key(table_name, columns, patient_ids)
        if self.enable_cache and cache_key in self._table_cache:
            if self.monitor:
                print(f"📦 Cache hit: {table_name}")
            cached_data = self._table_cache[cache_key]
            if optimize_dtypes:
                return self.table_ops.optimize_dtypes(cached_data.copy())
            else:
                return cached_data.copy()

        # Load with memory optimization
        try:
            # Determine optimal chunk size based on available memory
            optimal_chunk_size = self._calculate_optimal_chunk_size(table_name)

            # Load data
            data = self._load_with_memory_management(
                table_name,
                columns=columns,
                patient_ids=patient_ids,
                chunk_size=optimal_chunk_size,
                **kwargs
            )

            # Optimize dtypes if requested
            if optimize_dtypes and not data.empty:
                data = self.table_ops.optimize_dtypes(data)

            # Cache result if within memory limits
            if self.enable_cache and self._should_cache(data):
                self._cache_with_memory_management(cache_key, data)

            if self.monitor:
                self.monitor.log_memory_usage(f"Loaded {table_name}")
                self.monitor.cleanup_if_needed()

            return data

        except MemoryError:
            # Fallback to chunked loading
            warnings.warn(f"Memory error loading {table_name}, falling back to chunked loading")
            return self._load_table_chunked(
                table_name, columns=columns, patient_ids=patient_ids,
                chunk_size=self.monitor.config.chunk_size // 2,
                optimize_dtypes=optimize_dtypes, **kwargs
            )

    def _load_with_memory_management(self,
                                   table_name: str,
                                   *,
                                   columns: Optional[List[str]] = None,
                                   patient_ids: Optional[List[Any]] = None,
                                   chunk_size: Optional[int] = None,
                                   **kwargs) -> pd.DataFrame:
        """Load table with active memory management."""

        # Use parent's loading method but with memory monitoring
        try:
            # Disable caching temporarily to avoid memory issues
            original_cache_setting = self.enable_cache
            self.enable_cache = False

            # Load data
            data = super().load_table(
                table_name,
                columns=columns,
                patient_ids=patient_ids,
                **kwargs
            )

            # Restore cache setting
            self.enable_cache = original_cache_setting

            return data

        finally:
            # Always cleanup after loading
            gc.collect()
            self.monitor.cleanup_if_needed()

    def _load_table_chunked(self,
                          table_name: str,
                          *,
                          columns: Optional[List[str]] = None,
                          patient_ids: Optional[List[Any]] = None,
                          chunk_size: int = 5000,
                          optimize_dtypes: bool = True,
                          **kwargs) -> pd.DataFrame:
        """Load table in chunks to manage memory usage."""

        if self.monitor:
            self.monitor.log_memory_usage(f"Chunked load {table_name}")

        # For now, we'll implement a simple chunking strategy
        # This could be enhanced with database-specific chunking
        all_chunks = []

        try:
            # If patient_ids are specified, chunk by patients
            if patient_ids:
                for i in range(0, len(patient_ids), chunk_size):
                    chunk_patients = patient_ids[i:i + chunk_size]

                    chunk_data = self._load_with_memory_management(
                        table_name,
                        columns=columns,
                        patient_ids=chunk_patients,
                        **kwargs
                    )

                    if not chunk_data.empty:
                        if optimize_dtypes:
                            chunk_data = self.table_ops.optimize_dtypes(chunk_data)
                        all_chunks.append(chunk_data)

                    # Cleanup between chunks
                    gc.collect()
                    self.monitor.cleanup_if_needed()

            else:
                # Load without patient filtering, then chunk the result
                full_data = self._load_with_memory_management(
                    table_name, columns=columns, **kwargs
                )

                if not full_data.empty:
                    # Process in chunks if we have an ID column
                    id_col = self._get_id_column(table_name)
                    if id_col and id_col in full_data.columns:
                        unique_ids = full_data[id_col].unique()
                        for i in range(0, len(unique_ids), chunk_size):
                            chunk_ids = unique_ids[i:i + chunk_size]
                            mask = full_data[id_col].isin(chunk_ids)
                            chunk_data = full_data[mask].copy()

                            if optimize_dtypes:
                                chunk_data = self.table_ops.optimize_dtypes(chunk_data)
                            all_chunks.append(chunk_data)

                            # Cleanup
                            del chunk_data
                            gc.collect()
                    else:
                        # No ID column, return the (optimized) full data
                        if optimize_dtypes:
                            full_data = self.table_ops.optimize_dtypes(full_data)
                        return full_data

        except Exception as e:
            warnings.warn(f"Error during chunked loading of {table_name}: {e}")

        # Combine all chunks
        if all_chunks:
            result = pd.concat(all_chunks, ignore_index=True, sort=False)

            # Final optimization
            if optimize_dtypes:
                result = self.table_ops.optimize_dtypes(result)

            return result
        else:
            return pd.DataFrame()

    def _get_cache_key(self, table_name: str, columns: Optional[List[str]], patient_ids: Optional[List[Any]]) -> tuple:
        """Generate cache key with memory considerations."""
        columns_key = tuple(sorted(columns)) if columns else None
        patient_key = tuple(sorted(patient_ids[:10])) if patient_ids else None  # Limit key size
        return (table_name, columns_key, patient_key)

    def _should_cache(self, data: pd.DataFrame) -> bool:
        """Determine if data should be cached based on size and memory pressure."""

        if data.empty:
            return False

        # Check data size
        data_size = data.memory_usage(deep=True).sum()
        if data_size > self._cache_size_limit // 4:  # Don't cache if > 25% of cache limit
            return False

        # Check cache count
        if len(self._table_cache) >= self._max_cached_tables:
            return False

        # Check memory pressure
        pressure = self.monitor.check_memory_pressure()
        if pressure in ['HIGH', 'EMERGENCY']:
            return False

        return True

    def _cache_with_memory_management(self, cache_key: tuple, data: pd.DataFrame):
        """Cache data with memory management."""

        # Clear oldest cache entries if needed
        if len(self._table_cache) >= self._max_cached_tables:
            # Remove oldest entry
            oldest_key = next(iter(self._table_cache))
            del self._table_cache[oldest_key]

        # Add to cache
        self._table_cache[cache_key] = data.copy()

    def _calculate_optimal_chunk_size(self, table_name: str) -> int:
        """Calculate optimal chunk size based on available memory and table characteristics."""

        # Get current memory usage
        memory_info = self.monitor.get_memory_usage()
        available_memory = self.monitor.config.safe_memory - memory_info['process_mb'] * 1024 * 1024

        # Reserve memory for operations
        safe_memory = available_memory * 0.7  # Use 70% of available memory

        # Estimate table row size (this could be enhanced with actual statistics)
        estimated_row_size = 1000  # bytes per row (rough estimate)

        # Calculate chunk size
        chunk_size = max(1000, min(safe_memory // estimated_row_size, self.monitor.config.chunk_size))

        return int(chunk_size)

    def _get_id_column(self, table_name: str) -> Optional[str]:
        """Get the ID column for a table."""
        # This could be enhanced with table-specific logic
        common_id_columns = ['stay_id', 'subject_id', 'patientunitstayid', 'admissionid', 'patientid']

        # Try to get from table schema if available
        if hasattr(self.config, 'tables') and table_name in self.config.tables:
            table_config = self.config.tables[table_name]
            if 'id_cols' in table_config:
                return table_config['id_cols'][0] if table_config['id_cols'] else None

        # Fallback to common names
        return common_id_columns[0]  # Default to 'stay_id'

    def load_concepts_memory_efficient(self,
                                     concepts: List[str],
                                     patient_ids: Optional[List[Any]] = None,
                                     max_memory_mb: Optional[int] = None,
                                     **kwargs) -> Dict[str, pd.DataFrame]:
        """Load multiple concepts with memory efficiency."""

        if max_memory_mb is None:
            max_memory_mb = self.monitor.config.safe_memory // (1024 * 1024)

        if self.monitor:
            self.monitor.log_memory_usage("Multi-concept load start")

        results = {}

        # Load concepts one by one to manage memory
        for concept in concepts:
            try:
                if self.monitor:
                    self.monitor.cleanup_if_needed()

                # Load concept with memory optimization
                from .concept import ConceptResolver
                resolver = ConceptResolver(self, self.dictionary)

                concept_data = resolver.resolve_concept(
                    concept,
                    patient_ids=patient_ids,
                    **kwargs
                )

                if not concept_data.empty:
                    # Optimize memory usage
                    concept_data = self.table_ops.optimize_dtypes(concept_data)
                    results[concept] = concept_data

                # Check memory usage after each concept
                current_memory = self.monitor.get_memory_usage()['process_mb']
                if current_memory > max_memory_mb * 0.8:  # 80% of limit
                    warnings.warn(f"Memory limit approaching: {current_memory:.1f}MB / {max_memory_mb}MB")
                    break

            except Exception as e:
                warnings.warn(f"Failed to load concept {concept}: {e}")
                continue

        if self.monitor:
            self.monitor.log_memory_usage("Multi-concept load complete")

        return results

    def process_patients_streaming(self,
                                 patient_ids: List[Any],
                                 processor_func: callable,
                                 chunk_size: Optional[int] = None,
                                 **kwargs) -> List[Any]:
        """Process patients in streaming fashion to minimize memory usage."""

        if chunk_size is None:
            chunk_size = self.monitor.config.chunk_size

        if self.monitor:
            self.monitor.log_memory_usage("Streaming patient processing start")

        results = []

        for i in range(0, len(patient_ids), chunk_size):
            chunk_patients = patient_ids[i:i + chunk_size]

            try:
                # Process chunk
                chunk_result = processor_func(chunk_patients, **kwargs)
                if chunk_result is not None:
                    results.append(chunk_result)

                # Cleanup
                gc.collect()
                self.monitor.cleanup_if_needed()

                if self.monitor:
                    print(f"✅ Processed chunk {i//chunk_size + 1}/{(len(patient_ids) + chunk_size - 1)//chunk_size}")

            except Exception as e:
                warnings.warn(f"Error processing patient chunk {i//chunk_size + 1}: {e}")
                continue

        if self.monitor:
            self.monitor.log_memory_usage("Streaming patient processing complete")

        return results


def create_optimized_datasource(config, **kwargs) -> OptimizedDataSource:
    """Create an optimized data source for 16GB systems."""
    return OptimizedDataSource(config, **kwargs)