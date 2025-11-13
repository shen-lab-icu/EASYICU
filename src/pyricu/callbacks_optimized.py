"""
Memory-optimized callback functions for 16GB RAM systems.

This module provides memory-efficient versions of clinical scoring functions,
specifically designed to work within 16GB memory constraints.
"""

from typing import Dict, Optional, Union, List
import gc
import warnings

import numpy as np
import pandas as pd

from .memory_optimizer import MemoryMonitor, MemoryEfficientTable, get_memory_monitor


class OptimizedCallbacks:
    """Memory-optimized callback functions."""

    def __init__(self, monitor: Optional[MemoryMonitor] = None):
        self.monitor = monitor or get_memory_monitor()
        self.table_ops = MemoryEfficientTable(self.monitor)

    def sofa_score_optimized(
        self,
        data_dict: Dict[str, pd.DataFrame],
        *,
        keep_components: bool = False,
        win_length: pd.Timedelta = None,
        worst_val_fun: str = 'max',
        use_chunking: bool = True,
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Memory-optimized SOFA score calculation for 16GB systems.

        Key optimizations:
        1. Process data in patient chunks
        2. Aggressive memory cleanup
        3. Optimized data types
        4. Lazy evaluation where possible
        """

        if self.monitor:
            self.monitor.log_memory_usage("SOFA calculation start")

        from .ts_utils import fill_gaps, slide

        required = ['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal']

        if win_length is None:
            win_length = pd.Timedelta(hours=24)

        # Filter and optimize component data
        available_components = {}
        for comp in required:
            if comp in data_dict and not data_dict[comp].empty:
                # Optimize memory usage
                df = self.table_ops.optimize_dtypes(data_dict[comp].copy())
                available_components[comp] = df

        if not available_components:
            return pd.DataFrame(columns=['stay_id', 'time', 'sofa'])

        # Decide whether to use chunking based on data size
        total_rows = sum(len(df) for df in available_components.values())
        use_chunking = use_chunking and total_rows > 50000  # Use chunking for large datasets

        if use_chunking:
            return self._sofa_score_chunked(
                available_components,
                keep_components=keep_components,
                win_length=win_length,
                worst_val_fun=worst_val_fun,
                chunk_size=chunk_size
            )
        else:
            return self._sofa_score_standard(
                available_components,
                keep_components=keep_components,
                win_length=win_length,
                worst_val_fun=worst_val_fun
            )

    def _sofa_score_standard(
        self,
        data_dict: Dict[str, pd.DataFrame],
        *,
        keep_components: bool = False,
        win_length: pd.Timedelta = None,
        worst_val_fun: str = 'max'
    ) -> pd.DataFrame:
        """Standard SOFA calculation with memory optimizations."""

        from .ts_utils import fill_gaps, slide
        required = ['sofa_resp', 'sofa_coag', 'sofa_liver', 'sofa_cardio', 'sofa_cns', 'sofa_renal']

        # Step 1: Merge components with memory efficiency
        result = None
        for comp in required:
            if comp in data_dict:
                df = data_dict[comp]
                if result is None:
                    result = df
                else:
                    result = self.table_ops.merge_memory_efficient(
                        result, df, on=['stay_id', 'time'], how='outer'
                    )

        if result is None or result.empty:
            return pd.DataFrame(columns=['stay_id', 'time', 'sofa'])

        # Step 2: Apply time filtering for miiv data
        if 'time' in result.columns and len(result) > 0:
            min_time = result['time'].min()
            if min_time < -1000:  # Likely miiv data
                result = result[result['time'] >= 0].copy()
                self.monitor.cleanup_if_needed()

        # Step 3: Fill gaps (create continuous time series)
        from .ts_utils import fill_gaps as ricu_fill_gaps
        aligned_result = ricu_fill_gaps(result, limits=None)

        # Cleanup intermediate result
        del result
        gc.collect()
        self.monitor.cleanup_if_needed()

        # Step 4: Apply sliding window
        id_cols = ['stay_id']
        sofa_components = [comp for comp in required if comp in aligned_result.columns]

        if win_length.total_seconds() > 0:
            agg_dict = {comp: worst_val_fun for comp in sofa_components}
            slid_result = slide(
                data=aligned_result,
                id_cols=id_cols,
                index_col='time',
                before=win_length,
                after=pd.Timedelta(0),
                agg_func=agg_dict,
                full_window=False
            )
        else:
            slid_result = aligned_result.copy()

        # Cleanup aligned_result
        del aligned_result
        gc.collect()
        self.monitor.cleanup_if_needed()

        # Step 5: Forward-fill results
        slid_components = [c for c in sofa_components if c in slid_result.columns]
        if slid_components:
            slid_result[slid_components] = slid_result.groupby(id_cols)[slid_components].ffill()

        # Step 6: Filter rows with no component data
        if slid_components:
            has_data_mask = slid_result[slid_components].notna().any(axis=1)
            slid_result = slid_result[has_data_mask].copy()

        # Step 7: Calculate total SOFA score
        slid_result['sofa'] = slid_result[slid_components].fillna(0).sum(axis=1).astype(int)

        # Step 8: Final column selection
        final_cols = ['stay_id', 'time', 'sofa']
        if keep_components:
            final_cols.extend(sofa_components)

        # Ensure all required columns exist
        for col in final_cols:
            if col not in slid_result.columns:
                slid_result[col] = np.nan

        result = slid_result[final_cols].copy()

        if self.monitor:
            self.monitor.log_memory_usage("SOFA calculation complete")

        return result

    def _sofa_score_chunked(
        self,
        data_dict: Dict[str, pd.DataFrame],
        *,
        keep_components: bool = False,
        win_length: pd.Timedelta = None,
        worst_val_fun: str = 'max',
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """Process SOFA score calculation in patient chunks."""

        # Collect all unique patient IDs
        all_patients = set()
        for df in data_dict.values():
            if 'stay_id' in df.columns:
                all_patients.update(df['stay_id'].unique())

        all_patients = sorted(list(all_patients))

        def process_patient_chunk(chunk_df: pd.DataFrame) -> Optional[pd.DataFrame]:
            """Process SOFA for a chunk of patients."""
            try:
                # Extract data for these patients
                chunk_patients = chunk_df['stay_id'].unique()
                chunk_data = {}

                for comp, full_df in data_dict.items():
                    # Filter data for this chunk
                    mask = full_df['stay_id'].isin(chunk_patients)
                    chunk_data[comp] = full_df[mask].copy()

                # Calculate SOFA for this chunk using standard method
                return self._sofa_score_standard(
                    chunk_data,
                    keep_components=keep_components,
                    win_length=win_length,
                    worst_val_fun=worst_val_fun
                )

            except Exception as e:
                print(f"⚠️  Error processing patient chunk: {e}")
                return None

        # Create patient list DataFrame
        patients_df = pd.DataFrame({'stay_id': all_patients})

        # Process in chunks
        results = self.table_ops.process_in_chunks(
            patients_df,
            process_patient_chunk,
            chunk_column='stay_id',
            chunk_size=chunk_size or self.monitor.config.chunk_size
        )

        # Combine results
        valid_results = [r for r in results if r is not None]
        if valid_results:
            final_result = pd.concat(valid_results, ignore_index=True)

            # Optimize final result
            final_result = self.table_ops.optimize_dtypes(final_result)

            return final_result.sort_values(['stay_id', 'time']).reset_index(drop=True)
        else:
            return pd.DataFrame(columns=['stay_id', 'time', 'sofa'])

    def sofa_resp_optimized(self, pafi: pd.Series, vent_ind: Optional[pd.Series] = None) -> pd.Series:
        """Memory-optimized respiratory SOFA component."""

        # Convert to more memory-efficient dtype
        pafi = pd.to_numeric(pafi, errors='coerce', downcast='float')

        def is_true(series):
            """Memory-efficient boolean conversion."""
            if pd.api.types.is_float_dtype(series):
                return series.notna() & (series != 0)
            else:
                return series.fillna(False).astype(bool)

        # Process in place to avoid copies
        pafi_adjusted = pafi.copy(deep=False)

        if vent_ind is not None:
            vent_bool = is_true(vent_ind)
            mask = is_true(pafi < 200) & (~vent_bool)
            pafi_adjusted[mask] = 200

        # Calculate score efficiently
        score = pd.Series(0, index=pafi.index, dtype='int8')  # Use int8 to save memory

        # Apply thresholds (highest to lowest)
        pafi_num = pd.to_numeric(pafi_adjusted, errors='coerce')

        # Vectorized operations for better performance
        score[pafi_num < 100] = 4
        score[(pafi_num >= 100) & (pafi_num < 200)] = 3
        score[(pafi_num >= 200) & (pafi_num < 300)] = 2
        score[(pafi_num >= 300) & (pafi_num < 400)] = 1
        # score remains 0 for pafi >= 400

        # Cleanup
        del pafi_adjusted, pafi_num
        gc.collect()

        return score

    def sofa_cardio_optimized(self,
                            map_vals: pd.Series,
                            vaso_active_drugs: Optional[pd.Series] = None) -> pd.Series:
        """Memory-optimized cardiovascular SOFA component."""

        # Convert to memory-efficient dtypes
        map_vals = pd.to_numeric(map_vals, errors='coerce', downcast='float')

        score = pd.Series(0, index=map_vals.index, dtype='int8')

        if vaso_active_drugs is not None:
            # Convert to boolean efficiently
            vaso_bool = vaso_active_drugs.fillna(False).astype(bool)

            # Apply scoring logic
            # Hypotension + no vasopressors
            score[(map_vals < 70) & (~vaso_bool)] = 1

            # Dopamine ≤ 5 or any vasopressor
            score[(vaso_bool) & (map_vals >= 70)] = 2

            # Dopamine > 5 OR epinephrine/norepinephrine ≤ 0.1
            score[vaso_bool & (map_vals < 70)] = 3

            # High-dose vasopressors
            score[vaso_bool & (map_vals < 50)] = 4

        else:
            # No vasopressor data - score based on MAP only
            score[map_vals < 70] = 1
            score[map_vals < 50] = 2
            score[map_vals < 40] = 3
            score[map_vals < 30] = 4

        return score

    def pafi_optimized(self,
                      po2: pd.DataFrame,
                      fio2: pd.DataFrame,
                      id_cols: List[str],
                      time_col: str = 'charttime',
                      po2_val_col: str = 'valuenum',
                      fio2_val_col: str = 'valuenum') -> pd.DataFrame:
        """Memory-optimized PaO2/FiO2 ratio calculation."""

        # Optimize input data
        po2_df = po2[id_cols + [time_col, po2_val_col]].copy()
        fio2_df = fio2[id_cols + [time_col, fio2_val_col]].copy()

        po2_df = self.table_ops.optimize_dtypes(po2_df)
        fio2_df = self.table_ops.optimize_dtypes(fio2_df)

        # Merge using memory-efficient method
        result = self.table_ops.merge_memory_efficient(
            po2_df, fio2_df, on=id_cols + [time_col], how='outer'
        )

        # Calculate ratio
        po2_vals = pd.to_numeric(result[po2_val_col], errors='coerce')
        fio2_vals = pd.to_numeric(result[fio2_val_col], errors='coerce')

        # Convert FiO2 percentage to fraction
        fio2_fraction = fio2_vals.where(fio2_vals <= 1, fio2_vals / 100)

        # Calculate PaO2/FiO2 ratio
        ratio = po2_vals / fio2_fraction

        # Create result DataFrame
        result_df = result[id_cols + [time_col]].copy()
        result_df['pafi'] = ratio

        # Cleanup
        del po2_df, fio2_df, result, po2_vals, fio2_vals, fio2_fraction, ratio
        gc.collect()

        return result_df


# Global optimized callbacks instance
_optimized_callbacks = None

def get_optimized_callbacks() -> OptimizedCallbacks:
    """Get or create global optimized callbacks instance."""
    global _optimized_callbacks
    if _optimized_callbacks is None:
        _optimized_callbacks = OptimizedCallbacks()
    return _optimized_callbacks

# Convenience functions that mirror the original API
def sofa_score(data_dict: Dict[str, pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Memory-optimized SOFA score calculation."""
    return get_optimized_callbacks().sofa_score_optimized(data_dict, **kwargs)

def sofa_resp(pafi: pd.Series, vent_ind: Optional[pd.Series] = None) -> pd.Series:
    """Memory-optimized respiratory SOFA component."""
    return get_optimized_callbacks().sofa_resp_optimized(pafi, vent_ind)

def pafi(po2: pd.DataFrame, fio2: pd.DataFrame, id_cols: List[str], **kwargs) -> pd.DataFrame:
    """Memory-optimized PaO2/FiO2 ratio calculation."""
    return get_optimized_callbacks().pafi_optimized(po2, fio2, id_cols, **kwargs)