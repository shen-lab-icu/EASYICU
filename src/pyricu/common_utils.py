"""
Common utility functions for pyricu - unified implementations.

This module provides unified implementations of commonly used utility functions
to reduce code duplication and improve consistency across the codebase.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Union, Optional, List, Dict, Any, Callable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeriesUtils:
    """Unified Series utility functions."""

    @staticmethod
    def is_true(series: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
        """
        Unified is_true implementation that works across different data types.

        Replicates R's is_true: non-NA and True.

        Args:
            series: Input series or array

        Returns:
            Boolean series/array indicating True values

        Examples:
            >>> SeriesUtils.is_true(pd.Series([1, 0, np.nan, True, False]))
            [True, False, False, True, False]
        """
        if isinstance(series, pd.Series):
            if pd.api.types.is_float_dtype(series):
                # For float dtypes, check if not NaN and convert to bool
                return series.notna() & (series != 0)
            else:
                # For other types, use standard fillna
                return series.fillna(False).astype(bool)
        else:  # numpy array
            series = pd.Series(series)
            if pd.api.types.is_float_dtype(series):
                return series.notna() & (series != 0)
            else:
                return series.fillna(False).astype(bool).values

    @staticmethod
    def safe_convert_numeric(series: pd.Series,
                           downcast: bool = True,
                           errors: str = 'coerce') -> pd.Series:
        """
        Safely convert series to numeric with optional downcasting.

        Args:
            series: Input series
            downcast: Whether to downcast to smaller dtypes
            errors: How to handle errors ('coerce', 'raise', 'ignore')

        Returns:
            Numeric series
        """
        result = pd.to_numeric(series, errors=errors)

        if downcast and not result.empty:
            if pd.api.types.is_integer_dtype(result):
                result = pd.to_numeric(result, downcast='integer')
            elif pd.api.types.is_float_dtype(result):
                result = pd.to_numeric(result, downcast='float')

        return result


class DataFrameUtils:
    """Unified DataFrame utility functions."""

    @staticmethod
    def safe_copy(df: pd.DataFrame,
                  deep: bool = False,
                  memory_limit_mb: Optional[float] = None) -> pd.DataFrame:
        """
        Safe DataFrame copying with memory consideration.

        Args:
            df: DataFrame to copy
            deep: Whether to perform deep copy
            memory_limit_mb: Memory limit in MB for the copy operation

        Returns:
            Copied DataFrame

        Raises:
            MemoryError: If copy would exceed memory limit
        """
        if memory_limit_mb is not None:
            estimated_size = df.memory_usage(deep=True).sum() / (1024 * 1024)
            if estimated_size > memory_limit_mb:
                raise MemoryError(f"Copy would use {estimated_size:.1f}MB, exceeding limit {memory_limit_mb}MB")

        try:
            return df.copy(deep=deep)
        except MemoryError as e:
            if deep:
                logger.warning("Deep copy failed, trying shallow copy")
                return df.copy(deep=False)
            else:
                raise e

    @staticmethod
    def optimize_dtypes(df: pd.DataFrame,
                       category_threshold: float = 0.5) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes to reduce memory usage.

        Args:
            df: Input DataFrame
            category_threshold: Ratio of unique values to total values for converting to category

        Returns:
            DataFrame with optimized dtypes
        """
        original_memory = df.memory_usage(deep=True).sum()
        optimized_df = df.copy()

        # Convert object columns to category if they have low cardinality
        for col in optimized_df.select_dtypes(include=['object']).columns:
            num_unique = optimized_df[col].nunique()
            num_total = len(optimized_df[col])
            if num_total > 0 and num_unique / num_total < category_threshold:
                optimized_df[col] = optimized_df[col].astype('category')

        # Downcast numeric columns
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')

        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')

        # Log savings
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        savings = original_memory - optimized_memory
        if savings > 0:

        return optimized_df

    @staticmethod
    def memory_efficient_merge(left: pd.DataFrame,
                             right: pd.DataFrame,
                             on: List[str],
                             how: str = 'outer',
                             optimize_dtypes: bool = True) -> pd.DataFrame:
        """
        Memory-efficient DataFrame merge operation.

        Args:
            left: Left DataFrame
            right: Right DataFrame
            on: Column names to merge on
            how: Type of merge ('inner', 'outer', 'left', 'right')
            optimize_dtypes: Whether to optimize dtypes before merging

        Returns:
            Merged DataFrame
        """
        # Optimize dtypes before merge if requested
        if optimize_dtypes:
            left = DataFrameUtils.optimize_dtypes(left)
            right = DataFrameUtils.optimize_dtypes(right)

        # Perform merge
        try:
            return pd.merge(left, right, on=on, how=how)
        except MemoryError as e:
            logger.warning("Merge failed due to memory, trying alternative approach")

            # Alternative: merge in chunks if possible
            if 'stay_id' in on:
                return DataFrameUtils._merge_by_patients(left, right, on, how)
            else:
                raise e

    @staticmethod
    def _merge_by_patients(left: pd.DataFrame,
                          right: pd.DataFrame,
                          on: List[str],
                          how: str) -> pd.DataFrame:
        """Merge by patient IDs to handle memory constraints."""
        if 'stay_id' not in on:
            raise ValueError("Patient-based merge requires 'stay_id' in merge columns")

        # Get unique patient IDs
        all_patients = set(left['stay_id']).union(set(right['stay_id']))

        results = []
        chunk_size = 1000  # Process 1000 patients at a time

        for i in range(0, len(all_patients), chunk_size):
            chunk_patients = list(all_patients)[i:i + chunk_size]

            left_chunk = left[left['stay_id'].isin(chunk_patients)]
            right_chunk = right[right['stay_id'].isin(chunk_patients)]

            chunk_result = pd.merge(left_chunk, right_chunk, on=on, how=how)
            results.append(chunk_result)

        return pd.concat(results, ignore_index=True, sort=False)


class TimeSeriesUtils:
    """Unified time series utility functions."""

    @staticmethod
    def locf(series: pd.Series,
             max_gap: Optional[pd.Timedelta] = None,
             limit: Optional[int] = None) -> pd.Series:
        """
        Unified Last Observation Carried Forward implementation.

        Args:
            series: Input series
            max_gap: Maximum gap to fill
            limit: Maximum number of consecutive NaN values to fill

        Returns:
            Series with forward-filled values
        """
        if max_gap is not None:
            # For time-indexed series, respect the time gap
            if not isinstance(series.index, pd.DatetimeIndex):
                logger.warning("max_gap specified but series index is not datetime")

            # Group by consecutive NaN blocks and fill within max_gap
            nan_groups = (series.isna() != series.isna().shift()).cumsum()
            result = series.groupby(nan_groups).transform(
                lambda x: x.ffill(limit=limit) if x.isna().any() else x
            )

            # Respect max_gap by checking time differences
            if isinstance(series.index, pd.DatetimeIndex):
                time_diff = series.index.to_series().diff()
                large_gap_mask = time_diff > max_gap
                result[large_gap_mask] = np.nan

            return result
        else:
            return series.ffill(limit=limit)

    @staticmethod
    def safe_fill_gaps(df: pd.DataFrame,
                      id_cols: List[str],
                      time_col: str,
                      value_cols: List[str],
                      freq: str = 'H') -> pd.DataFrame:
        """
        Safe gap filling with memory consideration.

        Args:
            df: Input DataFrame
            id_cols: ID columns
            time_col: Time column
            value_cols: Value columns to fill
            freq: Frequency for time grid

        Returns:
            DataFrame with filled gaps
        """
        try:
            # Create complete time grid for each patient
            min_time = df[time_col].min()
            max_time = df[time_col].max()

            time_grid = pd.date_range(start=min_time, end=max_time, freq=freq)

            # Create complete index for each patient
            complete_index = []
            for _, group in df.groupby(id_cols):
                patient_times = pd.MultiIndex.from_product(
                    [group.iloc[0:1][id_cols].values[0], time_grid],
                    names=id_cols + [time_col]
                )
                complete_index.append(patient_times)

            if complete_index:
                full_index = pd.concat(complete_index)
                result = df.set_index(id_cols + [time_col]).reindex(full_index)
                result[value_cols] = result[value_cols].ffill()
                return result.reset_index()
            else:
                return df

        except MemoryError:
            logger.warning("Gap filling failed due to memory, using simpler approach")
            # Fallback: just forward fill existing data
            result = df.copy()
            result[value_cols] = result.groupby(id_cols)[value_cols].ffill()
            return result


class ValidationUtils:
    """Unified validation utility functions."""

    @staticmethod
    def validate_dataframe(df: pd.DataFrame,
                          required_cols: Optional[List[str]] = None,
                          id_cols: Optional[List[str]] = None,
                          time_col: Optional[str] = None) -> bool:
        """
        Validate DataFrame structure and content.

        Args:
            df: DataFrame to validate
            required_cols: Required columns
            id_cols: Expected ID columns
            time_col: Expected time column

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        if required_cols:
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

        if id_cols:
            for col in id_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing ID column: {col}")
                if df[col].isna().all():
                    raise ValueError(f"ID column {col} contains only NaN values")

        if time_col and time_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
                try:
                    pd.to_datetime(df[time_col])
                except Exception as e:
                    raise ValueError(f"Time column {time_col} cannot be converted to datetime: {e}")

        return True

    @staticmethod
    def check_memory_usage(df: pd.DataFrame,
                          operation: str = "unknown") -> Dict[str, float]:
        """
        Check and log DataFrame memory usage.

        Args:
            df: DataFrame to check
            operation: Operation name for logging

        Returns:
            Memory usage information
        """
        memory_info = {
            'total_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'numeric_mb': df.select_dtypes(include=[np.number]).memory_usage(deep=True).sum() / (1024 * 1024),
            'object_mb': df.select_dtypes(include=['object']).memory_usage(deep=True).sum() / (1024 * 1024),
            'rows': len(df),
            'cols': len(df.columns)
        }


        return memory_info


# Convenience functions for backward compatibility
def is_true(series: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """Backward compatible is_true function."""
    return SeriesUtils.is_true(series)

def safe_copy(df: pd.DataFrame, deep: bool = False) -> pd.DataFrame:
    """Backward compatible safe_copy function."""
    return DataFrameUtils.safe_copy(df, deep=deep)

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Backward compatible optimize_dtypes function."""
    return DataFrameUtils.optimize_dtypes(df)

def locf(series: pd.Series, max_gap: Optional[pd.Timedelta] = None) -> pd.Series:
    """Backward compatible locf function."""
    return TimeSeriesUtils.locf(series, max_gap=max_gap)