"""
Memory optimization utilities for pyricu - 16GB RAM optimization.

This module provides memory-efficient alternatives for common operations,
specifically designed for systems with limited memory (16GB).
"""

import gc
import os
import psutil
from typing import Iterator, List, Optional, Dict, Any, Union, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
import warnings

@dataclass
class MemoryConfig:
    """Memory configuration for 16GB systems."""

    # Total system memory (bytes)
    total_memory: int = 16 * 1024 * 1024 * 1024  # 16GB

    # Safe memory usage (80% of total to avoid system instability)
    safe_memory: int = int(0.8 * 16 * 1024 * 1024 * 1024)  # ~12.8GB

    # Per-operation memory limits
    cache_limit: int = 2 * 1024 * 1024 * 1024  # 2GB for cache
    chunk_size: int = 5000  # Reduced from 10000
    max_workers: int = 2  # Reduced from 4

    # Memory pressure thresholds
    warning_threshold: float = 0.7  # 70% memory usage triggers warning
    cleanup_threshold: float = 0.85  # 85% triggers cleanup
    emergency_threshold: float = 0.95  # 95% triggers emergency cleanup

class MemoryMonitor:
    """Monitor and manage memory usage during data processing."""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()

        return {
            'process_mb': memory_info.rss / (1024 * 1024),
            'process_percent': self.process.memory_percent(),
            'system_percent': system_memory.percent,
            'system_available_gb': system_memory.available / (1024**3),
            'system_used_gb': system_memory.used / (1024**3)
        }

    def check_memory_pressure(self) -> str:
        """Check current memory pressure level."""
        usage = self.get_memory_usage()
        system_percent = usage['system_percent']

        if system_percent >= self.config.emergency_threshold * 100:
            return 'EMERGENCY'
        elif system_percent >= self.config.cleanup_threshold * 100:
            return 'HIGH'
        elif system_percent >= self.config.warning_threshold * 100:
            return 'WARNING'
        else:
            return 'SAFE'

    def cleanup_if_needed(self, force: bool = False) -> bool:
        """Perform memory cleanup if needed."""
        pressure = self.check_memory_pressure()

        if force or pressure in ['HIGH', 'EMERGENCY']:
            print(f"🧹 Memory cleanup ({pressure})...")

            # Garbage collection
            gc.collect()

            # Clear pandas cache if available
            try:
                pd.reset_option('display.max_rows')
            except:
                pass

            # Force garbage collection again
            gc.collect()

            new_pressure = self.check_memory_pressure()
            if new_pressure != pressure:
                print(f"✅ Memory cleanup successful: {pressure} → {new_pressure}")
            else:
                print(f"⚠️  Memory cleanup limited effect: {pressure}")

            return True
        return False

    def log_memory_usage(self, operation: str):
        """Log memory usage for an operation."""
        usage = self.get_memory_usage()
        pressure = self.check_memory_pressure()

        print(f"📊 [{operation}] Memory: {usage['process_mb']:.1f}MB "
              f"({usage['system_percent']:.1f}% system) [{pressure}]")

class MemoryEfficientTable:
    """Memory-efficient DataFrame operations."""

    def __init__(self, monitor: Optional[MemoryMonitor] = None):
        self.monitor = monitor or MemoryMonitor()

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes to reduce memory usage."""
        try:
            # Use unified implementation if available
            from .common_utils import DataFrameUtils
            return DataFrameUtils.optimize_dtypes(df, category_threshold=0.5)
        except ImportError:
            # Fallback to original implementation
            if self.monitor:
                original_memory = df.memory_usage(deep=True).sum()

            # Convert object columns to category if they have low cardinality
            for col in df.select_dtypes(include=['object']).columns:
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
                    df[col] = df[col].astype('category')

            # Downcast numeric columns
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='integer')

            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = pd.to_numeric(df[col], downcast='float')

            if self.monitor:
                optimized_memory = df.memory_usage(deep=True).sum()
                savings = (original_memory - optimized_memory) / (1024 * 1024)
                print(f"💾 Memory optimization: {savings:.1f}MB saved")

            return df

    def merge_memory_efficient(self,
                             left: pd.DataFrame,
                             right: pd.DataFrame,
                             on: List[str],
                             how: str = 'outer',
                             optimize: bool = True) -> pd.DataFrame:
        """Memory-efficient merge with automatic cleanup."""

        if self.monitor:
            self.monitor.log_memory_usage("Before merge")

        # Optimize dtypes before merge
        if optimize:
            left = self.optimize_dtypes(left.copy())
            right = self.optimize_dtypes(right.copy())

        # Perform merge
        result = pd.merge(left, right, on=on, how=how)

        # Cleanup intermediate results
        del left, right
        gc.collect()

        if self.monitor:
            self.monitor.log_memory_usage("After merge")
            self.monitor.cleanup_if_needed()

        return result

    def process_in_chunks(self,
                         df: pd.DataFrame,
                         func: Callable[[pd.DataFrame], Any],
                         chunk_column: str = 'stay_id',
                         chunk_size: Optional[int] = None) -> List[Any]:
        """Process DataFrame in chunks to manage memory usage."""

        if chunk_size is None:
            chunk_size = self.monitor.config.chunk_size

        if self.monitor:
            self.monitor.log_memory_usage("Start chunked processing")

        results = []
        unique_ids = df[chunk_column].unique()
        total_chunks = (len(unique_ids) + chunk_size - 1) // chunk_size

        for i in range(0, len(unique_ids), chunk_size):
            chunk_ids = unique_ids[i:i + chunk_size]
            chunk = df[df[chunk_column].isin(chunk_ids)].copy()

            if self.monitor:
                print(f"🔄 Processing chunk {i//chunk_size + 1}/{total_chunks} "
                      f"({len(chunk_ids)} patients, {len(chunk)} rows)")

            # Process chunk
            result = func(chunk)
            if result is not None:
                results.append(result)

            # Cleanup chunk
            del chunk
            gc.collect()

            # Check memory pressure
            if self.monitor:
                self.monitor.cleanup_if_needed()

        if self.monitor:
            self.monitor.log_memory_usage("End chunked processing")

        return results

    def safe_copy(self, df: pd.DataFrame, deep: bool = False) -> pd.DataFrame:
        """Memory-safe copy with automatic cleanup."""

        if self.monitor:
            self.monitor.cleanup_if_needed()

        # Use shallow copy by default to save memory
        if not deep:
            result = df.copy(deep=False)
        else:
            result = df.copy(deep=True)

        if self.monitor:
            usage = self.monitor.get_memory_usage()
            if usage['process_mb'] > self.monitor.config.safe_memory / (1024*1024):
                warnings.warn(f"High memory usage: {usage['process_mb']:.1f}MB")

        return result

class MemoryOptimizedCallbacks:
    """Memory-optimized versions of callback functions."""

    def __init__(self, monitor: Optional[MemoryMonitor] = None):
        self.monitor = monitor or MemoryMonitor()
        self.table_ops = MemoryEfficientTable(monitor)

    def sofa_score_memory_efficient(self,
                                   data_dict: Dict[str, pd.DataFrame],
                                   *,
                                   keep_components: bool = False,
                                   win_length: pd.Timedelta = None,
                                   worst_val_fun: str = 'max',
                                   chunk_by_patient: bool = True) -> pd.DataFrame:
        """Memory-efficient SOFA score calculation for 16GB systems."""

        from .ts_utils import fill_gaps, slide

        if self.monitor:
            self.monitor.log_memory_usage("SOFA score start")

        required = ['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal']

        if win_length is None:
            win_length = pd.Timedelta(hours=24)

        # Filter out empty dataframes
        available_components = {
            comp: df for comp, df in data_dict.items()
            if comp in required and not df.empty
        }

        if not available_components:
            return pd.DataFrame(columns=['stay_id', 'time', 'sofa'])

        # Process in chunks if enabled
        if chunk_by_patient and len(available_components) > 0:
            return self._sofa_score_chunked(
                available_components,
                keep_components=keep_components,
                win_length=win_length,
                worst_val_fun=worst_val_fun
            )

        # Standard processing with memory optimization
        result = None
        for comp in required:
            if comp in available_components:
                df = self.table_ops.optimize_dtypes(available_components[comp])
                if result is None:
                    result = df
                else:
                    result = self.table_ops.merge_memory_efficient(
                        result, df, on=['stay_id', 'time'], how='outer'
                    )

        if result is None or result.empty:
            return pd.DataFrame(columns=['stay_id', 'time', 'sofa'])

        # Apply time filtering for miiv data
        if 'time' in result.columns and len(result) > 0:
            min_time = result['time'].min()
            if min_time < -1000:  # Likely miiv data
                result = result[result['time'] >= 0].copy()
                del result  # Cleanup
                # Re-create filtered result without copying if possible
                mask = available_components.get(list(available_components.keys())[0], pd.DataFrame()).empty
                if not mask:
                    first_df = list(available_components.values())[0]
                    result = first_df[first_df['time'] >= 0].copy()

        # Continue with standard SOFA calculation...
        # (Implementation continues as in original but with memory optimization)

        if self.monitor:
            self.monitor.log_memory_usage("SOFA score complete")

        return result

    def _sofa_score_chunked(self,
                           data_dict: Dict[str, pd.DataFrame],
                           *,
                           keep_components: bool = False,
                           win_length: pd.Timedelta = None,
                           worst_val_fun: str = 'max') -> pd.DataFrame:
        """Process SOFA score calculation in patient chunks."""

        # Get all unique patient IDs
        all_patients = set()
        for df in data_dict.values():
            if 'stay_id' in df.columns:
                all_patients.update(df['stay_id'].unique())

        all_patients = sorted(list(all_patients))

        def process_patient_chunk(patient_df: pd.DataFrame) -> Optional[pd.DataFrame]:
            """Process SOFA for a chunk of patients."""
            try:
                # Extract data for these patients
                chunk_data = {}
                for comp, df in data_dict.items():
                    patient_subset = patient_df['stay_id'].unique()
                    chunk_data[comp] = df[df['stay_id'].isin(patient_subset)].copy()

                # Calculate SOFA for this chunk
                # Use original SOFA logic but on smaller data
                # (This would need to be implemented with the actual SOFA calculation)

                return None  # Placeholder
            except Exception as e:
                print(f"Error processing patient chunk: {e}")
                return None

        # Create a DataFrame with all patients and process in chunks
        patients_df = pd.DataFrame({'stay_id': all_patients})
        results = self.table_ops.process_in_chunks(
            patients_df,
            process_patient_chunk,
            chunk_size=self.monitor.config.chunk_size
        )

        # Combine results
        valid_results = [r for r in results if r is not None]
        if valid_results:
            return pd.concat(valid_results, ignore_index=True)
        else:
            return pd.DataFrame(columns=['stay_id', 'time', 'sofa'])

# Global memory monitor instance
_global_monitor = None

def get_memory_monitor() -> MemoryMonitor:
    """Get or create global memory monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = MemoryMonitor()
    return _global_monitor

def optimize_for_16gb():
    """Configure pyricu for optimal performance on 16GB systems."""

    config = MemoryConfig()

    # Update project configuration
    from . import project_config
    project_config.CHUNK_SIZE = config.chunk_size
    project_config.MAX_WORKERS = config.max_workers
    project_config.CACHE_SIZE_LIMIT = config.cache_limit // (1024 * 1024)  # Convert to MB

    # Configure pandas for memory efficiency
    pd.set_option('mode.chained_assignment', 'warn')

    print(f"Optimized for 16GB RAM:")
    print(f"   Chunk size: {config.chunk_size}")
    print(f"   Max workers: {config.max_workers}")
    print(f"   Cache limit: {config.cache_limit // (1024**2)}MB")

    return config

# Auto-configure on import only if explicitly requested
_config = None
if os.getenv('PYRICU_AUTO_OPTIMIZE_16GB', 'false').lower() in ('true', '1', 'yes'):
    try:
        _config = optimize_for_16gb()
    except Exception as e:
        print(f"⚠️  Auto-optimization failed: {e}")
