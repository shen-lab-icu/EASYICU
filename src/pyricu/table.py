"""High level table abstractions for ICU datasets (R ricu tbl-class.R).

Provides id_tbl, ts_tbl, and win_tbl classes that wrap pandas DataFrames
with metadata for ICU data handling, corresponding to R ricu's table classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Union, Dict, Callable
from copy import deepcopy

import pandas as pd
import numpy as np

@dataclass
class ICUTable:
    """Wrap a :class:`pandas.DataFrame` together with column metadata."""

    data: pd.DataFrame
    id_columns: List[str] = field(default_factory=list)
    index_column: Optional[str] = None
    value_column: Optional[str] = None
    unit_column: Optional[str] = None
    time_columns: List[str] = field(default_factory=list)
    _owns_data: bool = field(default=False, repr=False)  # 🚀 标记是否拥有数据

    def __post_init__(self) -> None:
        # 🚀 优化：避免不必要的copy，大部分情况下数据是只读的
        # 如果需要修改，调用者应该传入_owns_data=True或显式调用.copy()
        pass  # 移除强制copy
        self._validate_columns(self.id_columns, required=False)
        self._validate_columns(self.time_columns, required=False)
        if self.index_column:
            self._validate_columns([self.index_column])
        if self.value_column:
            self._validate_columns([self.value_column])
        if self.unit_column:
            self._validate_columns([self.unit_column])

    def _validate_columns(self, columns: Iterable[str], required: bool = True) -> None:
        for column in columns:
            if column not in self.data.columns:
                if required:
                    raise KeyError(f"Column '{column}' not present in table")
            elif required is False:
                continue

    def copy(self) -> "ICUTable":
        return ICUTable(
            data=self.data.copy(),
            id_columns=list(self.id_columns),
            index_column=self.index_column,
            value_column=self.value_column,
            unit_column=self.unit_column,
            time_columns=list(self.time_columns),
        )

    def dataframe(self) -> pd.DataFrame:
        """Return the underlying :class:`DataFrame`."""
        return self.data

    def select_ids(self, values: Iterable) -> "ICUTable":
        """Filter the table to a subset of identifier values."""
        if not self.id_columns:
            return self.copy()
        filtered = self.data[self.data[self.id_columns[0]].isin(list(values))]
        return ICUTable(
            data=filtered,
            id_columns=self.id_columns,
            index_column=self.index_column,
            value_column=self.value_column,
            unit_column=self.unit_column,
            time_columns=self.time_columns,
        )

    def iter_groups(self) -> Iterator[pd.DataFrame]:
        """Iterate over groups keyed by the ID columns."""
        if not self.id_columns:
            yield self.data
            return
        grouped = self.data.groupby(self.id_columns, sort=False, dropna=False)
        for _, frame in grouped:
            yield frame

    def to_wide(self, column_name: Optional[str] = None) -> pd.DataFrame:
        """Return a wide-format representation using the ID and index columns."""
        if not self.id_columns or not self.index_column or not self.value_column:
            raise ValueError(
                "Wide conversion requires id, index, and value column metadata"
            )

        concept_col = column_name or self.value_column
        wide = (
            self.data.set_index(self.id_columns + [self.index_column])[self.value_column]
            .unstack(self.index_column)
            .rename(concept_col)
        )
        return wide

# ============================================================================
# Enhanced table classes (R ricu tbl-class.R)
# ============================================================================

class IdTbl:
    """ID table - static patient data (R ricu id_tbl).
    
    Wraps a pandas DataFrame with metadata about ID columns.
    Corresponds to R's id_tbl class for static patient data.
    
    Attributes:
        data: The underlying DataFrame
        id_vars: List of ID column names
        
    Examples:
        >>> df = pd.DataFrame({'patient_id': [1, 2], 'age': [65, 72]})
        >>> tbl = IdTbl(df, id_vars=['patient_id'])
    """
    
    def __init__(
        self, 
        data: pd.DataFrame, 
        id_vars: Union[str, List[str]],
        by_ref: bool = False
    ):
        """Initialize IdTbl.
        
        Args:
            data: DataFrame with patient data
            id_vars: ID column name(s)
            by_ref: If True, use data by reference; if False, copy
        """
        if isinstance(id_vars, str):
            id_vars = [id_vars]
        
        self.id_vars = id_vars
        
        if by_ref:
            self.data = data
        else:
            self.data = data.copy()
        
        # Validate
        self._validate()
        
        # Reorder columns: ID columns first
        self._reorder_columns()
        
        # Remove NA in ID columns
        self._remove_na_ids()
    
    def _validate(self):
        """Validate table structure."""
        # Check ID columns exist
        missing = set(self.id_vars) - set(self.data.columns)
        if missing:
            raise ValueError(f"ID columns not found: {missing}")
        
        # Check unique column names
        if len(self.data.columns) != len(set(self.data.columns)):
            raise ValueError("Column names must be unique")
    
    def _reorder_columns(self):
        """Move ID columns to front."""
        other_cols = [c for c in self.data.columns if c not in self.id_vars]
        self.data = self.data[self.id_vars + other_cols]
    
    def _remove_na_ids(self):
        """Remove rows with NA in ID columns."""
        self.data = self.data.dropna(subset=self.id_vars)
    
    def meta_vars(self) -> List[str]:
        """Get metadata variable names (ID columns)."""
        return self.id_vars
    
    def data_vars(self) -> List[str]:
        """Get data variable names (non-ID columns)."""
        return [c for c in self.data.columns if c not in self.id_vars]
    
    def copy(self) -> "IdTbl":
        """Create a copy of this table."""
        return IdTbl(self.data.copy(), self.id_vars.copy(), by_ref=False)
    
    def __len__(self) -> int:
        """Number of rows."""
        return len(self.data)
    
    def __repr__(self) -> str:
        """美化输出 (R ricu print.id_tbl)."""
        n_rows = len(self.data)
        n_cols = len(self.data.columns)
        n_ids = self.data[self.id_vars[0]].nunique() if self.id_vars else 0
        
        header = f"# An id_tbl: {n_rows:,} x {n_cols} [{n_ids:,} unique IDs]"
        return f"{header}\n{self.data.head(10).to_string()}"
    
    def __str__(self) -> str:
        """字符串表示."""
        return self.__repr__()
    
    def summary(self) -> pd.DataFrame:
        """统计摘要 (R ricu summary.id_tbl).
        
        Returns:
            DataFrame with summary statistics for each data variable
        """
        summary_data = []
        
        for col in self.data_vars():
            if pd.api.types.is_numeric_dtype(self.data[col]):
                summary_data.append({
                    'variable': col,
                    'type': str(self.data[col].dtype),
                    'count': self.data[col].count(),
                    'missing': self.data[col].isna().sum(),
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'min': self.data[col].min(),
                    'q25': self.data[col].quantile(0.25),
                    'median': self.data[col].median(),
                    'q75': self.data[col].quantile(0.75),
                    'max': self.data[col].max(),
                })
            else:
                summary_data.append({
                    'variable': col,
                    'type': str(self.data[col].dtype),
                    'count': self.data[col].count(),
                    'missing': self.data[col].isna().sum(),
                    'unique': self.data[col].nunique(),
                })
        
        return pd.DataFrame(summary_data)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to regular DataFrame."""
        return self.data.copy()
    
    def as_ts_tbl(
        self,
        index_var: str,
        interval: Optional[pd.Timedelta] = None
    ) -> "TsTbl":
        """Convert IdTbl to TsTbl (R ricu as_ts_tbl).
        
        Args:
            index_var: Time index column name
            interval: Time interval (auto-detected if None)
            
        Returns:
            TsTbl object
            
        Examples:
            >>> id_tbl = IdTbl(df, id_vars=['patient_id'])
            >>> ts_tbl = id_tbl.as_ts_tbl('time')
        """
        return TsTbl(
            self.data,
            self.id_vars,
            index_var=index_var,
            interval=interval,
            by_ref=False
        )
    
    def as_win_tbl(
        self,
        index_var: str,
        dur_var: str,
        interval: Optional[pd.Timedelta] = None
    ) -> "WinTbl":
        """Convert IdTbl to WinTbl (R ricu as_win_tbl).
        
        Args:
            index_var: Time index column name
            dur_var: Duration column name
            interval: Time interval (auto-detected if None)
            
        Returns:
            WinTbl object
            
        Examples:
            >>> id_tbl = IdTbl(df, id_vars=['patient_id'])
            >>> win_tbl = id_tbl.as_win_tbl('start_time', 'duration')
        """
        return WinTbl(
            self.data,
            self.id_vars,
            index_var=index_var,
            dur_var=dur_var,
            interval=interval,
            by_ref=False
        )

class TsTbl(IdTbl):
    """Time series table (R ricu ts_tbl).
    
    Extends IdTbl with time index column and interval metadata.
    Corresponds to R's ts_tbl class for grouped time series data.
    
    Attributes:
        data: The underlying DataFrame
        id_vars: List of ID column names
        index_var: Time index column name
        interval: Time step as pd.Timedelta
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'patient_id': [1, 1, 2, 2],
        ...     'time': pd.timedelta_range(0, periods=4, freq='1H'),
        ...     'hr': [80, 85, 90, 95]
        ... })
        >>> tbl = TsTbl(df, id_vars=['patient_id'], index_var='time')
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        id_vars: Union[str, List[str]],
        index_var: Optional[str] = None,
        interval: Optional[pd.Timedelta] = None,
        by_ref: bool = False
    ):
        """Initialize TsTbl.
        
        Args:
            data: DataFrame with time series data
            id_vars: ID column name(s)
            index_var: Time index column (auto-detected if None)
            interval: Time step (auto-detected if None)
            by_ref: If True, use data by reference
        """
        # Auto-detect index_var if not provided
        if index_var is None:
            index_var = self._detect_index_var(data)
        
        self.index_var = index_var
        
        # Initialize parent
        super().__init__(data, id_vars, by_ref)
        
        # Auto-detect interval if not provided
        if interval is None:
            interval = self._detect_interval()
        
        self.interval = interval
    
    @staticmethod
    def _detect_index_var(data: pd.DataFrame) -> str:
        """Auto-detect time index column."""
        time_cols = [
            c for c in data.columns
            if pd.api.types.is_timedelta64_dtype(data[c]) or
               pd.api.types.is_datetime64_any_dtype(data[c])
        ]
        
        if len(time_cols) == 0:
            raise ValueError("No timedelta or datetime column found for index_var")
        elif len(time_cols) > 1:
            raise ValueError(f"Multiple time columns found: {time_cols}. Please specify index_var.")
        
        return time_cols[0]
    
    def _detect_interval(self) -> pd.Timedelta:
        """Auto-detect time interval from data."""
        # Check if index is numeric (eICU-style) or temporal
        is_numeric = pd.api.types.is_numeric_dtype(self.data[self.index_var])
        
        # Get unique time differences
        diffs = []
        for _, group in self.data.groupby(self.id_vars):
            if len(group) > 1:
                sorted_times = group[self.index_var].sort_values()
                group_diffs = sorted_times.diff().dropna()
                diffs.extend(group_diffs.tolist())
        
        if not diffs:
            # Default to 1 hour if no differences found
            return pd.Timedelta(hours=1)
        
        # Return minimum non-zero difference
        if is_numeric:
            # For numeric times (hours), convert to Timedelta
            min_diff = min(d for d in diffs if d > 0)
            return pd.Timedelta(hours=min_diff)
        else:
            # For temporal types, already Timedelta
            min_diff = min(d for d in diffs if d > pd.Timedelta(0))
            return min_diff
    
    def _validate(self):
        """Validate time series table structure."""
        super()._validate()
        
        # Check index column exists
        if self.index_var not in self.data.columns:
            raise ValueError(f"Index column '{self.index_var}' not found")
        
        # Check index is not an ID column
        if self.index_var in self.id_vars:
            raise ValueError(f"Index column '{self.index_var}' cannot be an ID column")
        
        # Check index column type
        # Allow: datetime, timedelta, or numeric (for eICU-style offset in hours)
        is_temporal = (pd.api.types.is_timedelta64_dtype(self.data[self.index_var]) or
                      pd.api.types.is_datetime64_any_dtype(self.data[self.index_var]))
        is_numeric = pd.api.types.is_numeric_dtype(self.data[self.index_var])
        
        if not (is_temporal or is_numeric):
            raise ValueError(
                f"Index column '{self.index_var}' must be datetime, timedelta, or numeric. "
                f"Got: {self.data[self.index_var].dtype}"
            )
    
    def _reorder_columns(self):
        """Move ID and index columns to front."""
        other_cols = [c for c in self.data.columns 
                     if c not in self.id_vars + [self.index_var]]
        self.data = self.data[self.id_vars + [self.index_var] + other_cols]
    
    def _remove_na_ids(self):
        """Remove rows with NA in ID or index columns."""
        self.data = self.data.dropna(subset=self.id_vars + [self.index_var])
    
    def meta_vars(self) -> List[str]:
        """Get metadata variable names (ID + index columns)."""
        return self.id_vars + [self.index_var]
    
    def time_vars(self) -> List[str]:
        """Get time variable names."""
        return [self.index_var]
    
    def copy(self) -> "TsTbl":
        """Create a copy of this table."""
        return TsTbl(
            self.data.copy(), 
            self.id_vars.copy(),
            self.index_var,
            self.interval,
            by_ref=False
        )
    
    def __repr__(self) -> str:
        """美化输出 (R ricu print.ts_tbl)."""
        n_rows = len(self.data)
        n_cols = len(self.data.columns)
        n_ids = self.data[self.id_vars[0]].nunique() if self.id_vars else 0
        
        # Time range
        time_range = ""
        if self.index_var:
            min_time = self.data[self.index_var].min()
            max_time = self.data[self.index_var].max()
            if pd.api.types.is_timedelta64_dtype(self.data[self.index_var]):
                time_range = f" ({min_time} to {max_time})"
            else:
                time_range = f" ({min_time} to {max_time})"
        
        header = f"# A ts_tbl: {n_rows:,} x {n_cols} [{n_ids:,} unique IDs]{time_range}"
        interval_info = f"# Interval: {self.interval}"
        
        return f"{header}\n{interval_info}\n{self.data.head(10).to_string()}"
    
    def __str__(self) -> str:
        """字符串表示."""
        return self.__repr__()
    
    def as_win_tbl(
        self,
        dur_var: str
    ) -> "WinTbl":
        """Convert TsTbl to WinTbl (R ricu as_win_tbl).
        
        Args:
            dur_var: Duration column name
            
        Returns:
            WinTbl object
            
        Examples:
            >>> ts_tbl = TsTbl(df, id_vars=['patient_id'], index_var='time')
            >>> win_tbl = ts_tbl.as_win_tbl('duration')
        """
        return WinTbl(
            self.data,
            self.id_vars,
            index_var=self.index_var,
            dur_var=dur_var,
            interval=self.interval,
            by_ref=False
        )

class WinTbl(TsTbl):
    """Window table (R ricu win_tbl).
    
    Extends TsTbl with duration/validity interval for each measurement.
    Corresponds to R's win_tbl class for windowed time series data.
    
    Attributes:
        data: The underlying DataFrame
        id_vars: List of ID column names
        index_var: Time index column name
        dur_var: Duration column name
        interval: Time step as pd.Timedelta
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'patient_id': [1, 1],
        ...     'time': pd.timedelta_range(0, periods=2, freq='1H'),
        ...     'duration': [pd.Timedelta(hours=2), pd.Timedelta(hours=3)],
        ...     'medication': ['drug_a', 'drug_b']
        ... })
        >>> tbl = WinTbl(df, id_vars=['patient_id'], index_var='time', dur_var='duration')
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        id_vars: Union[str, List[str]],
        index_var: Optional[str] = None,
        dur_var: Optional[str] = None,
        interval: Optional[pd.Timedelta] = None,
        by_ref: bool = False
    ):
        """Initialize WinTbl.
        
        Args:
            data: DataFrame with windowed time series data
            id_vars: ID column name(s)
            index_var: Time index column (auto-detected if None)
            dur_var: Duration column (auto-detected if None)
            interval: Time step (auto-detected if None)
            by_ref: If True, use data by reference
        """
        # Auto-detect dur_var if not provided
        if dur_var is None:
            dur_var = self._detect_dur_var(data, index_var)
        
        self.dur_var = dur_var
        
        # Initialize parent
        super().__init__(data, id_vars, index_var, interval, by_ref)
    
    @staticmethod
    def _detect_dur_var(data: pd.DataFrame, index_var: Optional[str] = None) -> str:
        """Auto-detect duration column."""
        time_cols = [
            c for c in data.columns
            if pd.api.types.is_timedelta64_dtype(data[c])
        ]
        
        # Exclude index_var if already determined
        if index_var:
            time_cols = [c for c in time_cols if c != index_var]
        
        if len(time_cols) == 0:
            raise ValueError("No timedelta column found for dur_var")
        elif len(time_cols) > 1:
            raise ValueError(f"Multiple duration columns found: {time_cols}. Please specify dur_var.")
        
        return time_cols[0]
    
    def _validate(self):
        """Validate window table structure."""
        super()._validate()
        
        # Check duration column exists
        if self.dur_var not in self.data.columns:
            raise ValueError(f"Duration column '{self.dur_var}' not found")
        
        # Check duration is not ID or index column
        if self.dur_var in self.id_vars + [self.index_var]:
            raise ValueError(f"Duration column '{self.dur_var}' cannot be an ID or index column")
        
        # Check duration column type
        # Allow: timedelta or numeric (for eICU-style duration in hours)
        is_timedelta = pd.api.types.is_timedelta64_dtype(self.data[self.dur_var])
        is_numeric = pd.api.types.is_numeric_dtype(self.data[self.dur_var])
        
        if not (is_timedelta or is_numeric):
            raise ValueError(
                f"Duration column '{self.dur_var}' must be timedelta or numeric. "
                f"Got: {self.data[self.dur_var].dtype}"
            )
    
    def _reorder_columns(self):
        """Move ID, index, and duration columns to front."""
        other_cols = [c for c in self.data.columns 
                     if c not in self.id_vars + [self.index_var, self.dur_var]]
        self.data = self.data[self.id_vars + [self.index_var, self.dur_var] + other_cols]
    
    def meta_vars(self) -> List[str]:
        """Get metadata variable names (ID + index + duration columns)."""
        return self.id_vars + [self.index_var, self.dur_var]
    
    def time_vars(self) -> List[str]:
        """Get time variable names."""
        return [self.index_var, self.dur_var]
    
    def copy(self) -> "WinTbl":
        """Create a copy of this table."""
        return WinTbl(
            self.data.copy(),
            self.id_vars.copy(),
            self.index_var,
            self.dur_var,
            self.interval,
            by_ref=False
        )
    
    def __repr__(self) -> str:
        """美化输出 (R ricu print.win_tbl)."""
        n_rows = len(self.data)
        n_cols = len(self.data.columns)
        n_ids = self.data[self.id_vars[0]].nunique() if self.id_vars else 0
        
        header = f"# A win_tbl: {n_rows:,} x {n_cols} [{n_ids:,} unique IDs]"
        interval_info = f"# Interval: {self.interval}"
        dur_info = f"# Duration var: {self.dur_var}"
        
        return f"{header}\n{interval_info}\n{dur_info}\n{self.data.head(10).to_string()}"
    
    def __str__(self) -> str:
        """字符串表示."""
        return self.__repr__()

class PvalTbl(TsTbl):
    """P-value table with statistical metadata (R ricu pval_tbl).
    
    Extends TsTbl to include p-value or statistical significance information.
    Used for storing results of statistical tests or significance thresholds.
    
    Attributes:
        data: The underlying DataFrame
        id_vars: List of ID column names
        index_var: Time index column name
        interval: Time step as pd.Timedelta
        pval_var: P-value column name
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'patient_id': [1, 1, 2, 2],
        ...     'time': pd.timedelta_range(0, periods=4, freq='1H'),
        ...     'test_stat': [2.5, 3.1, 1.8, 2.9],
        ...     'pvalue': [0.01, 0.002, 0.07, 0.004]
        ... })
        >>> tbl = PvalTbl(df, id_vars=['patient_id'], index_var='time', pval_var='pvalue')
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        id_vars: Union[str, List[str]],
        index_var: Optional[str] = None,
        pval_var: Optional[str] = None,
        interval: Optional[pd.Timedelta] = None,
        by_ref: bool = False
    ):
        """Initialize PvalTbl.
        
        Args:
            data: DataFrame with time series and p-value data
            id_vars: ID column name(s)
            index_var: Time index column (auto-detected if None)
            pval_var: P-value column name (auto-detected if None)
            interval: Time step (auto-detected if None)
            by_ref: If True, use data by reference
        """
        # Auto-detect pval_var if not provided
        if pval_var is None:
            pval_var = self._detect_pval_var(data)
        
        self.pval_var = pval_var
        
        # Initialize parent
        super().__init__(data, id_vars, index_var, interval, by_ref)
    
    @staticmethod
    def _detect_pval_var(data: pd.DataFrame) -> str:
        """Auto-detect p-value column."""
        # Look for common p-value column names
        pval_candidates = [
            c for c in data.columns
            if any(name in c.lower() for name in ['pval', 'p_val', 'pvalue', 'p.value'])
        ]
        
        if len(pval_candidates) == 0:
            # Try to find numeric columns with values in [0, 1]
            numeric_cols = data.select_dtypes(include='number').columns
            for col in numeric_cols:
                if data[col].between(0, 1).all():
                    return col
            raise ValueError("No p-value column found")
        elif len(pval_candidates) > 1:
            raise ValueError(f"Multiple p-value columns found: {pval_candidates}. Please specify pval_var.")
        
        return pval_candidates[0]
    
    def _validate(self):
        """Validate p-value table structure."""
        super()._validate()
        
        # Check p-value column exists
        if self.pval_var not in self.data.columns:
            raise ValueError(f"P-value column '{self.pval_var}' not found")
        
        # Check p-value column is numeric
        if not pd.api.types.is_numeric_dtype(self.data[self.pval_var]):
            raise ValueError(f"P-value column '{self.pval_var}' must be numeric")
        
        # Warn if values outside [0, 1]
        if not self.data[self.pval_var].between(0, 1).all():
            import warnings
            warnings.warn(f"P-value column '{self.pval_var}' contains values outside [0, 1]")
    
    def significant(self, alpha: float = 0.05) -> pd.Series:
        """Get boolean mask of statistically significant results.
        
        Args:
            alpha: Significance level (default 0.05)
            
        Returns:
            Boolean Series indicating significant results (p < alpha)
        """
        return self.data[self.pval_var] < alpha
    
    def filter_significant(self, alpha: float = 0.05) -> "PvalTbl":
        """Filter to only statistically significant results.
        
        Args:
            alpha: Significance level (default 0.05)
            
        Returns:
            New PvalTbl with only significant results
        """
        mask = self.significant(alpha)
        filtered = self.data[mask].reset_index(drop=True)
        
        return PvalTbl(
            filtered,
            self.id_vars.copy(),
            self.index_var,
            self.pval_var,
            self.interval,
            by_ref=False
        )
    
    def copy(self) -> "PvalTbl":
        """Create a copy of this table."""
        return PvalTbl(
            self.data.copy(),
            self.id_vars.copy(),
            self.index_var,
            self.pval_var,
            self.interval,
            by_ref=False
        )
    
    def __repr__(self) -> str:
        """美化输出 (R ricu print.pval_tbl)."""
        n_rows = len(self.data)
        n_cols = len(self.data.columns)
        n_ids = self.data[self.id_vars[0]].nunique() if self.id_vars else 0
        n_sig = self.significant().sum()
        
        header = f"# A pval_tbl: {n_rows:,} x {n_cols} [{n_ids:,} unique IDs]"
        pval_info = f"# P-value var: {self.pval_var} ({n_sig} significant at α=0.05)"
        
        return f"{header}\n{pval_info}\n{self.data.head(10).to_string()}"
    
    def __str__(self) -> str:
        """字符串表示."""
        return self.__repr__()

# ============================================================================
# Table utility functions (R ricu tbl-base.R)
# ============================================================================

def rbind_tbl(*tables: Union[IdTbl, pd.DataFrame], 
              use_names: bool = True, 
              fill: bool = False) -> Union[IdTbl, pd.DataFrame]:
    """Row-bind tables (R ricu rbind_id_tbl).
    
    Args:
        *tables: Tables to combine
        use_names: Whether to use column names for matching
        fill: Whether to fill missing columns with NA
        
    Returns:
        Combined table
    """
    # Extract DataFrames
    dfs = []
    for tbl in tables:
        if isinstance(tbl, (IdTbl, TsTbl, WinTbl)):
            dfs.append(tbl.data)
        else:
            dfs.append(tbl)
    
    # Combine
    if fill:
        result = pd.concat(dfs, ignore_index=True, sort=False)
    else:
        result = pd.concat(dfs, ignore_index=True)
    
    # Return same type as first input
    if isinstance(tables[0], WinTbl):
        return WinTbl(
            result,
            tables[0].id_vars,
            tables[0].index_var,
            tables[0].dur_var,
            tables[0].interval
        )
    elif isinstance(tables[0], TsTbl):
        return TsTbl(
            result,
            tables[0].id_vars,
            tables[0].index_var,
            tables[0].interval
        )
    elif isinstance(tables[0], IdTbl):
        return IdTbl(result, tables[0].id_vars)
    else:
        return result

def cbind_tbl(*tables: Union[IdTbl, pd.DataFrame],
              check_names: bool = False) -> Union[IdTbl, pd.DataFrame]:
    """Column-bind tables (R ricu cbind_id_tbl).
    
    Args:
        *tables: Tables to combine
        check_names: Whether to check for duplicate column names
        
    Returns:
        Combined table
    """
    # Extract DataFrames
    dfs = []
    for tbl in tables:
        if isinstance(tbl, (IdTbl, TsTbl, WinTbl)):
            dfs.append(tbl.data)
        else:
            dfs.append(tbl)
    
    # Combine
    result = pd.concat(dfs, axis=1)
    
    # Check for duplicates if requested
    if check_names and len(result.columns) != len(set(result.columns)):
        raise ValueError("Duplicate column names found")
    
    # Return same type as first input
    if isinstance(tables[0], (IdTbl, TsTbl, WinTbl)):
        return type(tables[0])(result, tables[0].id_vars)
    else:
        return result

def merge_lst(tables: List[Union[IdTbl, pd.DataFrame]], 
             by: Optional[List[str]] = None,
             how: str = 'outer') -> Union[IdTbl, pd.DataFrame]:
    """Merge list of tables (R ricu merge_lst).
    
    Args:
        tables: List of tables to merge
        by: Columns to merge on (if None, use all common columns)
        how: Type of merge ('inner', 'outer', 'left', 'right')
        
    Returns:
        Merged table
    """
    if not tables:
        return pd.DataFrame()
    
    if len(tables) == 1:
        return tables[0]
    
    # Extract DataFrames and determine merge keys
    dfs = []
    for tbl in tables:
        if isinstance(tbl, (IdTbl, TsTbl, WinTbl)):
            dfs.append(tbl.data)
            if by is None:
                by = tbl.meta_vars()
        else:
            dfs.append(tbl)
    
    # Merge all
    result = dfs[0]
    for df in dfs[1:]:
        result = pd.merge(result, df, on=by, how=how)
    
    # Return same type as first input
    if isinstance(tables[0], WinTbl):
        return WinTbl(
            result,
            tables[0].id_vars,
            tables[0].index_var,
            tables[0].dur_var,
            tables[0].interval
        )
    elif isinstance(tables[0], TsTbl):
        return TsTbl(
            result,
            tables[0].id_vars,
            tables[0].index_var,
            tables[0].interval
        )
    elif isinstance(tables[0], IdTbl):
        return IdTbl(result, tables[0].id_vars)
    else:
        return result

# ============================================================================
# Table validation functions (R ricu tbl-base.R)
# ============================================================================

def is_id_tbl(x) -> bool:
    """Check if object is an IdTbl (R ricu is_id_tbl).
    
    Args:
        x: Object to check
        
    Returns:
        True if x is an IdTbl, False otherwise
    """
    return isinstance(x, IdTbl)

def is_ts_tbl(x) -> bool:
    """Check if object is a TsTbl (R ricu is_ts_tbl).
    
    Args:
        x: Object to check
        
    Returns:
        True if x is a TsTbl, False otherwise
    """
    return isinstance(x, TsTbl)

def is_win_tbl(x) -> bool:
    """Check if object is a WinTbl (R ricu is_win_tbl).
    
    Args:
        x: Object to check
        
    Returns:
        True if x is a WinTbl, False otherwise
    """
    return isinstance(x, WinTbl)

def is_icu_tbl(x) -> bool:
    """Check if object is any ICU table type.
    
    Args:
        x: Object to check
        
    Returns:
        True if x is an ICUTable, IdTbl, TsTbl, or WinTbl
    """
    return isinstance(x, (ICUTable, IdTbl, TsTbl, WinTbl))

def has_time_cols(x) -> bool:
    """Check if table has time columns.
    
    Args:
        x: Table to check
        
    Returns:
        True if table has time index or duration columns
    """
    if isinstance(x, (TsTbl, WinTbl)):
        return True
    elif isinstance(x, ICUTable):
        return x.index_column is not None or len(x.time_columns) > 0
    else:
        return False

def validate_tbl_structure(
    data: pd.DataFrame,
    id_vars: List[str],
    index_var: Optional[str] = None,
    dur_var: Optional[str] = None,
) -> None:
    """Validate table structure (R ricu validate_tbl).
    
    Args:
        data: DataFrame to validate
        id_vars: Expected ID columns
        index_var: Expected index column
        dur_var: Expected duration column
        
    Raises:
        ValueError: If validation fails
    """
    # Check ID columns exist
    missing = set(id_vars) - set(data.columns)
    if missing:
        raise ValueError(f"Missing ID columns: {missing}")
    
    # Check index column if specified
    if index_var is not None and index_var not in data.columns:
        raise ValueError(f"Missing index column: {index_var}")
    
    # Check duration column if specified
    if dur_var is not None and dur_var not in data.columns:
        raise ValueError(f"Missing duration column: {dur_var}")
    
    # Check for NA in ID columns
    if data[id_vars].isna().any().any():
        raise ValueError("ID columns cannot contain NA values")
    
    # Check for duplicate column names
    if len(data.columns) != len(set(data.columns)):
        raise ValueError("Column names must be unique")

def id_vars(x) -> List[str]:
    """Get ID variable names from a table (R ricu id_vars).
    
    Args:
        x: ICU table
        
    Returns:
        List of ID column names
    """
    if isinstance(x, (IdTbl, TsTbl, WinTbl)):
        return x.id_vars
    elif isinstance(x, ICUTable):
        return list(x.id_columns)
    else:
        return []

def index_var(x) -> Optional[str]:
    """Get index variable name from a table (R ricu index_var).
    
    Args:
        x: ICU table
        
    Returns:
        Index column name or None
    """
    if isinstance(x, (TsTbl, WinTbl)):
        return x.index_var
    elif isinstance(x, ICUTable):
        return x.index_column
    else:
        return None

def dur_var(x) -> Optional[str]:
    """Get duration variable name from a table (R ricu dur_var).
    
    Args:
        x: ICU table
        
    Returns:
        Duration column name or None
    """
    if isinstance(x, WinTbl):
        return x.dur_var
    else:
        return None

def meta_vars(x) -> List[str]:
    """Get all metadata variable names from a table (R ricu meta_vars).
    
    Args:
        x: ICU table
        
    Returns:
        List of metadata column names (IDs, index, duration)
    """
    if isinstance(x, (IdTbl, TsTbl, WinTbl)):
        return x.meta_vars()
    elif isinstance(x, ICUTable):
        cols = list(x.id_columns)
        if x.index_column:
            cols.append(x.index_column)
        return cols
    else:
        return []

def data_vars(x) -> List[str]:
    """Get data variable names from a table (R ricu data_vars).
    
    Args:
        x: ICU table
        
    Returns:
        List of non-metadata column names
    """
    if isinstance(x, (IdTbl, TsTbl, WinTbl)):
        return x.data_vars()
    elif isinstance(x, ICUTable):
        meta = meta_vars(x)
        return [col for col in x.data.columns if col not in meta]
    else:
        return []

# ============================================================================
# ID type conversion functions (R ricu tbl-utils.R change_id)
# ============================================================================

def upgrade_id(
    data: pd.DataFrame,
    id_map: pd.DataFrame,
    from_id: str,
    to_id: str,
    keep_old_id: bool = False,
) -> pd.DataFrame:
    """Upgrade ID type to a higher level (R ricu upgrade_id).
    
    Converts IDs to a higher-level identifier (e.g., hadm_id -> icustay_id).
    This is a one-to-many relationship.
    
    Args:
        data: Input DataFrame with lower-level IDs
        id_map: Mapping DataFrame with both ID columns
        from_id: Current ID column name (lower level)
        to_id: Target ID column name (higher level)
        keep_old_id: Whether to keep the original ID column
        
    Returns:
        DataFrame with upgraded IDs
        
    Examples:
        >>> # Upgrade from hadm_id to icustay_id
        >>> vitals = pd.DataFrame({'hadm_id': [1, 1, 2], 'hr': [80, 85, 90]})
        >>> mapping = pd.DataFrame({'hadm_id': [1, 1, 2], 'icustay_id': [10, 11, 20]})
        >>> upgrade_id(vitals, mapping, 'hadm_id', 'icustay_id')
        # Result: 3 rows become 4 rows (hadm_id=1 duplicated for 2 stays)
    """
    # Validate inputs
    if from_id not in data.columns:
        raise ValueError(f"Column '{from_id}' not found in data")
    if from_id not in id_map.columns or to_id not in id_map.columns:
        raise ValueError(f"Columns '{from_id}' and '{to_id}' must be in id_map")
    
    # Get unique mapping (remove duplicates in id_map)
    mapping = id_map[[from_id, to_id]].drop_duplicates()
    
    # Merge to add new ID
    result = data.merge(mapping, on=from_id, how='left')
    
    # Optionally remove old ID
    if not keep_old_id:
        result = result.drop(columns=[from_id])
    
    return result

def downgrade_id(
    data: pd.DataFrame,
    id_map: pd.DataFrame,
    from_id: str,
    to_id: str,
    agg_funcs: Optional[Dict[str, Union[str, Callable]]] = None,
    keep_old_id: bool = False,
) -> pd.DataFrame:
    """Downgrade ID type to a lower level (R ricu downgrade_id).
    
    Converts IDs to a lower-level identifier (e.g., icustay_id -> hadm_id).
    This is a many-to-one relationship requiring aggregation.
    
    Args:
        data: Input DataFrame with higher-level IDs
        id_map: Mapping DataFrame with both ID columns
        from_id: Current ID column name (higher level)
        to_id: Target ID column name (lower level)
        agg_funcs: Dictionary mapping column names to aggregation functions
                   (default: first for non-numeric, mean for numeric)
        keep_old_id: Whether to keep the original ID column
        
    Returns:
        DataFrame with downgraded IDs and aggregated data
        
    Examples:
        >>> # Downgrade from icustay_id to hadm_id
        >>> vitals = pd.DataFrame({
        ...     'icustay_id': [10, 11, 20],
        ...     'hr': [80, 85, 90],
        ...     'temp': [36.5, 37.0, 36.8]
        ... })
        >>> mapping = pd.DataFrame({
        ...     'icustay_id': [10, 11, 20],
        ...     'hadm_id': [1, 1, 2]
        ... })
        >>> downgrade_id(vitals, mapping, 'icustay_id', 'hadm_id',
        ...              agg_funcs={'hr': 'mean', 'temp': 'mean'})
        # Result: 3 rows become 2 rows (stays 10 and 11 merged into hadm 1)
    """
    # Validate inputs
    if from_id not in data.columns:
        raise ValueError(f"Column '{from_id}' not found in data")
    if from_id not in id_map.columns or to_id not in id_map.columns:
        raise ValueError(f"Columns '{from_id}' and '{to_id}' must be in id_map")
    
    # Get unique mapping
    mapping = id_map[[from_id, to_id]].drop_duplicates()
    
    # Merge to add new ID
    result = data.merge(mapping, on=from_id, how='left')
    
    # Determine columns to aggregate
    group_cols = [to_id]
    if keep_old_id:
        group_cols.append(from_id)
    
    data_cols = [col for col in result.columns if col not in [from_id, to_id]]
    
    # Build aggregation dict
    if agg_funcs is None:
        agg_funcs = {}
        for col in data_cols:
            if pd.api.types.is_numeric_dtype(result[col]):
                agg_funcs[col] = 'mean'
            else:
                agg_funcs[col] = 'first'
    
    # Apply aggregation if needed
    if not keep_old_id:
        result = result.drop(columns=[from_id])
        result = result.groupby(to_id, as_index=False).agg(agg_funcs)
    else:
        result = result.groupby([to_id, from_id], as_index=False).agg(agg_funcs)
    
    return result

def change_id(
    data: pd.DataFrame,
    id_map: pd.DataFrame,
    from_id: str,
    to_id: str,
    keep_old_id: bool = False,
    agg_funcs: Optional[Dict[str, Union[str, Callable]]] = None,
) -> pd.DataFrame:
    """Change ID type (auto-detect upgrade vs downgrade) (R ricu change_id).
    
    Automatically determines whether to upgrade or downgrade based on
    the cardinality of the ID mapping.
    
    Args:
        data: Input DataFrame
        id_map: Mapping DataFrame with both ID columns
        from_id: Current ID column name
        to_id: Target ID column name
        keep_old_id: Whether to keep the original ID column
        agg_funcs: Aggregation functions (for downgrade only)
        
    Returns:
        DataFrame with changed IDs
        
    Examples:
        >>> # Auto-detect direction
        >>> change_id(data, mapping, 'hadm_id', 'icustay_id')
    """
    # Check cardinality to determine direction
    mapping = id_map[[from_id, to_id]].drop_duplicates()
    
    from_unique = mapping[from_id].nunique()
    to_unique = mapping[to_id].nunique()
    
    if to_unique > from_unique:
        # Upgrade (one-to-many)
        return upgrade_id(data, id_map, from_id, to_id, keep_old_id)
    elif to_unique < from_unique:
        # Downgrade (many-to-one)
        return downgrade_id(data, id_map, from_id, to_id, agg_funcs, keep_old_id)
    else:
        # One-to-one mapping
        mapping_dict = dict(zip(mapping[from_id], mapping[to_id]))
        result = data.copy()
        result[to_id] = result[from_id].map(mapping_dict)
        
        if not keep_old_id:
            result = result.drop(columns=[from_id])
        
        return result

def rbind_lst(
    tables: List[Union[pd.DataFrame, ICUTable, IdTbl, TsTbl, WinTbl]],
    id_var_opts: Optional[List[str]] = None,
) -> Union[pd.DataFrame, ICUTable]:
    """Row-bind a list of tables (R ricu rbind_lst).
    
    Combines multiple tables by rows, handling metadata appropriately.
    
    Args:
        tables: List of tables to combine
        id_var_opts: Optional list of ID variable options for conflict resolution
        
    Returns:
        Combined table
        
    Examples:
        >>> t1 = pd.DataFrame({'id': [1, 2], 'val': [10, 20]})
        >>> t2 = pd.DataFrame({'id': [3, 4], 'val': [30, 40]})
        >>> rbind_lst([t1, t2])
    """
    if not tables:
        return pd.DataFrame()
    
    # Extract DataFrames
    dfs = []
    metadata = None
    
    for table in tables:
        if isinstance(table, pd.DataFrame):
            dfs.append(table)
        elif isinstance(table, (IdTbl, TsTbl, WinTbl)):
            dfs.append(table.data)
            if metadata is None:
                metadata = {
                    'id_vars': table.id_vars,
                    'type': type(table)
                }
                if isinstance(table, (TsTbl, WinTbl)):
                    metadata['index_var'] = table.index_var
                    metadata['interval'] = table.interval
                if isinstance(table, WinTbl):
                    metadata['dur_var'] = table.dur_var
        elif isinstance(table, ICUTable):
            dfs.append(table.data)
            if metadata is None:
                metadata = {
                    'id_columns': table.id_columns,
                    'index_column': table.index_column,
                    'value_column': table.value_column,
                }
    
    # Concatenate
    result_df = pd.concat(dfs, ignore_index=True)
    
    # Reconstruct with metadata if available
    if metadata and 'type' in metadata:
        table_type = metadata['type']
        if table_type == IdTbl:
            return IdTbl(result_df, metadata['id_vars'])
        elif table_type == TsTbl:
            return TsTbl(
                result_df,
                metadata['id_vars'],
                metadata['index_var'],
                metadata['interval']
            )
        elif table_type == WinTbl:
            return WinTbl(
                result_df,
                metadata['id_vars'],
                metadata['index_var'],
                metadata['dur_var'],
                metadata['interval']
            )
    elif metadata and 'id_columns' in metadata:
        return ICUTable(
            data=result_df,
            id_columns=metadata['id_columns'],
            index_column=metadata.get('index_column'),
            value_column=metadata.get('value_column'),
        )
    
    return result_df

def rename_cols(
    data: Union[pd.DataFrame, ICUTable, IdTbl, TsTbl, WinTbl],
    old_names: Union[str, List[str]],
    new_names: Union[str, List[str]],
    by_ref: bool = False,
) -> Union[pd.DataFrame, ICUTable]:
    """Rename columns, updating metadata attributes (R ricu rename_cols).
    
    Args:
        data: Input table
        old_names: Old column name(s)
        new_names: New column name(s)
        by_ref: Whether to modify in place
        
    Returns:
        Table with renamed columns
        
    Examples:
        >>> rename_cols(df, 'old_name', 'new_name')
        >>> rename_cols(df, ['a', 'b'], ['x', 'y'])
    """
    if isinstance(old_names, str):
        old_names = [old_names]
    if isinstance(new_names, str):
        new_names = [new_names]
    
    if len(old_names) != len(new_names):
        raise ValueError("old_names and new_names must have the same length")
    
    rename_dict = dict(zip(old_names, new_names))
    
    if isinstance(data, pd.DataFrame):
        if by_ref:
            data.rename(columns=rename_dict, inplace=True)
            return data
        else:
            return data.rename(columns=rename_dict)
    
    elif isinstance(data, (IdTbl, TsTbl, WinTbl)):
        if not by_ref:
            data = data.copy()
        
        # Rename in DataFrame
        data.data.rename(columns=rename_dict, inplace=True)
        
        # Update metadata
        for i, old_name in enumerate(old_names):
            new_name = new_names[i]
            
            # Update ID vars
            if old_name in data.id_vars:
                idx = data.id_vars.index(old_name)
                data.id_vars[idx] = new_name
            
            # Update index var
            if isinstance(data, (TsTbl, WinTbl)) and data.index_var == old_name:
                data.index_var = new_name
            
            # Update duration var
            if isinstance(data, WinTbl) and data.dur_var == old_name:
                data.dur_var = new_name
        
        return data
    
    elif isinstance(data, ICUTable):
        if not by_ref:
            data = data.copy()
        
        # Rename in DataFrame
        data.data.rename(columns=rename_dict, inplace=True)
        
        # Update metadata
        for i, old_name in enumerate(old_names):
            new_name = new_names[i]
            
            # Update ID columns
            if old_name in data.id_columns:
                idx = data.id_columns.index(old_name)
                data.id_columns[idx] = new_name
            
            # Update other columns
            if data.index_column == old_name:
                data.index_column = new_name
            if data.value_column == old_name:
                data.value_column = new_name
            if data.unit_column == old_name:
                data.unit_column = new_name
            if old_name in data.time_columns:
                idx = data.time_columns.index(old_name)
                data.time_columns[idx] = new_name
        
        return data
    
    else:
        raise TypeError(f"Unsupported type: {type(data)}")

def rm_cols(
    data: Union[pd.DataFrame, ICUTable, IdTbl, TsTbl, WinTbl],
    cols: Union[str, List[str]],
    skip_absent: bool = False,
    by_ref: bool = False,
) -> Union[pd.DataFrame, ICUTable]:
    """Remove columns from a table (R ricu rm_cols).
    
    Args:
        data: Input table
        cols: Column name(s) to remove
        skip_absent: If True, ignore columns that don't exist
        by_ref: Whether to modify in place
        
    Returns:
        Table with columns removed
        
    Examples:
        >>> rm_cols(df, 'unwanted_col')
        >>> rm_cols(df, ['col1', 'col2'], skip_absent=True)
    """
    if isinstance(cols, str):
        cols = [cols]
    
    if isinstance(data, pd.DataFrame):
        if skip_absent:
            cols = [c for c in cols if c in data.columns]
        
        if by_ref:
            data.drop(columns=cols, inplace=True, errors='ignore' if skip_absent else 'raise')
            return data
        else:
            return data.drop(columns=cols, errors='ignore' if skip_absent else 'raise')
    
    elif isinstance(data, (IdTbl, TsTbl, WinTbl, ICUTable)):
        if not by_ref:
            data = data.copy()
        
        if skip_absent:
            cols = [c for c in cols if c in data.data.columns]
        
        data.data.drop(columns=cols, inplace=True, errors='ignore' if skip_absent else 'raise')
        return data
    
    else:
        raise TypeError(f"Unsupported type: {type(data)}")

# 便捷函数用于加载表
def load_table(src: str, table_name: str, **kwargs) -> pd.DataFrame:
    """
    加载表 - 便捷函数
    
    Args:
        src: 数据源名称
        table_name: 表名
        **kwargs: 其他参数传递给 ICUDataSource.load_table
        
    Returns:
        DataFrame
        
    Examples:
        >>> df = load_table('mimic_demo', 'patients')
    """
    from .attach import data
    
    # 首先尝试从全局附加的数据源获取
    if data.is_attached(src):
        data_source = data.get_source(src)
    else:
        # 如果没有附加，尝试从配置创建
        from .config import load_src_cfg
        config = load_src_cfg(src)
        from .datasource import ICUDataSource
        data_source = ICUDataSource(config)
    
    icu_table = data_source.load_table(table_name, **kwargs)
    return icu_table.data
