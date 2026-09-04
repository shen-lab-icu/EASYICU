#!/usr/bin/env python3
"""
⚠️ DEPRECATED: 该模块的功能应该迁移到测试框架 ⚠️

参考ricu的设计哲学：
- ricu没有单独的"feature_compare"模块
- 验证逻辑在tests/testthat/test-*.R中
- 使用标准测试框架（testthat）而非独立脚本

easyicu应该遵循相同模式：
- 核心对比逻辑 → tests/helpers.py (pytest fixtures)
- 验证断言 → tests/core/test_ricu_alignment.py (pytest测试用例)
- CLI工具 → multi_db_feature_alignment.py可保留作为便捷脚本

迁移状态（2026-06 校正：此前 1/2 标 ✅ 但文件并不存在，属虚假声明，已修正）：
1. ✅ tests/helpers.py 已真实创建 - 提供 load_ricu_csv / pooled_median /
   median_of_medians / FakeSource / assert_series_close 等。
2. ✅ tests/core/test_ricu_alignment.py 已真实创建 - 覆盖池化 median 决策与时间单位，
   并预留 ricu CSV 黄金基准用例 (fixture 缺失时自动跳过)。
3. ⏳ 逐步将依赖 feature_compare 的脚本迁移到使用 tests/helpers。
4. ⏳ 最终删除此文件。

临时保留原因：
- multi_db_feature_alignment.py 当前仍依赖此模块。
- 需要时间将所有验证逻辑迁移到 pytest。

---

原文档：
Reusable feature comparison utilities between ricu CSV outputs and easyicu.

This module hosts the heavy lifting that used to live in the legacy
``compare_ricu_easyicu.py`` script so that other tools (and downstream users)
can leverage the same logic programmatically. The thin CLI wrapper now just
parses arguments and delegates to :func:`main`.
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import load_concepts
from .runtime.project_config import (
    DEFAULT_PATIENTS_AUMC,
    DEFAULT_PATIENTS_EICU,
    DEFAULT_PATIENTS_HIRID,
    DEFAULT_PATIENTS_MIIV,
    PRODUCTION_DATA_AUMC,
    PRODUCTION_DATA_EICU,
    PRODUCTION_DATA_HIRID,
    PRODUCTION_DATA_PATH,
)
from .runtime.runtime_defaults import resolve_loader_defaults

# Import shared configuration from modules_config
from .modules_config import (
    FeatureModule,
    MODULES,
    SOFA_COMPONENT_DEPENDENCIES,
    SUPPORTED_DATABASES,
    ID_CANDIDATES,
    TIME_CANDIDATES,
)

# Note: FeatureModule, MODULES, MODULES_BY_NAME, ALL_CONCEPTS, SOFA_COMPONENT_DEPENDENCIES,
# SUPPORTED_DATABASES, DATABASE_ID_COLUMNS, ID_CANDIDATES, TIME_CANDIDATES are now imported
# from modules_config.py

FIO2_ITEMIDS = [223835, 50816]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_R_DATA_ROOT = REPO_ROOT / "ricu_data"

ALIGNMENT_ENV_KEYS = {
    "chunk": ("EASYICU_ALIGN_CHUNK", "EASYICU_CHUNK_SIZE"),
    "parallel": ("EASYICU_ALIGN_PARALLEL", "EASYICU_PARALLEL_WORKERS"),
    "concept": ("EASYICU_ALIGN_CONCEPT", "EASYICU_CONCEPT_WORKERS"),
    "backend": ("EASYICU_ALIGN_BACKEND", "EASYICU_PARALLEL_BACKEND"),
}

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATHS: Dict[str, Path] = {
    "miiv": PRODUCTION_DATA_PATH,
    "eicu": PRODUCTION_DATA_EICU,
    "aumc": PRODUCTION_DATA_AUMC,
    "hirid": PRODUCTION_DATA_HIRID,
}

DEFAULT_PATIENTS: Dict[str, List[int]] = {
    "miiv": DEFAULT_PATIENTS_MIIV,
    "eicu": DEFAULT_PATIENTS_EICU,
    "aumc": DEFAULT_PATIENTS_AUMC,
    "hirid": DEFAULT_PATIENTS_HIRID or [],
}

PATIENT_ID_SOURCES: Dict[str, tuple[str, str]] = {
    "miiv": ("icustays.parquet", "stay_id"),
    "eicu": ("patient.parquet", "patientunitstayid"),
    "aumc": ("admissions.parquet", "admissionid"),
    "hirid": ("general_table.csv", "patientid"),  # 🔧 FIX: 使用正确的表名
}

@dataclass
class SeriesStats:
    rows: int
    non_null: int
    mean: Optional[float]
    maximum: Optional[float]
    first_signal: Optional[float]

    @property
    def coverage(self) -> float:
        return (self.non_null / self.rows) if self.rows else 0.0

class FeatureComparison:
    """Holds normalised ricu/easyicu time series for a module."""

    def __init__(
        self,
        module: FeatureModule,
        ricu_series: Dict[str, pd.DataFrame],
        easyicu_series: Dict[str, pd.DataFrame],
    ):
        self.module = module
        self.ricu_series = ricu_series
        self.easyicu_series = easyicu_series

    def available_concepts(self, source: str) -> Iterable[str]:
        mapping = self.easyicu_series if source == "easyicu" else self.ricu_series
        return mapping.keys()

    def compute_metrics(self) -> Dict[str, Dict[str, object]]:
        metrics: Dict[str, Dict[str, object]] = {}
        names = sorted(set(self.ricu_series.keys()) | set(self.easyicu_series.keys()))
        for name in names:
            ricu_stats = self._stats(self.ricu_series.get(name))
            py_stats = self._stats(self.easyicu_series.get(name))
            entry = {
                "ricu_rows": ricu_stats.rows,
                "easyicu_rows": py_stats.rows,
                "ricu_coverage": ricu_stats.coverage,
                "easyicu_coverage": py_stats.coverage,
                "coverage_gap": ricu_stats.coverage - py_stats.coverage,
                "ricu_mean": ricu_stats.mean,
                "easyicu_mean": py_stats.mean,
                "mean_diff": (py_stats.mean - ricu_stats.mean)
                if (ricu_stats.mean is not None and py_stats.mean is not None)
                else None,
                "ricu_max": ricu_stats.maximum,
                "easyicu_max": py_stats.maximum,
                "ricu_first": ricu_stats.first_signal,
                "easyicu_first": py_stats.first_signal,
            }
            if py_stats.rows == 0 and ricu_stats.rows > 0:
                entry["status"] = "missing:easyicu"
            elif py_stats.rows > 0 and ricu_stats.rows == 0:
                entry["status"] = "missing:ricu"
            elif ricu_stats.non_null > 0 and py_stats.non_null == 0:
                entry["status"] = "missing:easyicu"
            else:
                entry["status"] = "ok"
            metrics[name] = entry
        return metrics

    @staticmethod
    def _stats(df: Optional[pd.DataFrame]) -> SeriesStats:
        if df is None or df.empty:
            return SeriesStats(rows=0, non_null=0, mean=None, maximum=None, first_signal=None)

        values = df["value"]
        if isinstance(values, pd.DataFrame) or (hasattr(values, "ndim") and values.ndim > 1):
            return SeriesStats(rows=len(df), non_null=0, mean=None, maximum=None, first_signal=None)
        if not isinstance(values, pd.Series):
            values = pd.Series(values)
        try:
            numeric = pd.to_numeric(values, errors="coerce")
        except Exception:
            numeric = pd.Series(np.nan, index=values.index)
        numeric_valid = numeric.notna().sum() > 0
        mean = float(numeric.mean()) if numeric_valid else None
        maximum = float(numeric.max()) if numeric_valid else None

        first_signal = None
        if "time" in df.columns:
            if numeric_valid:
                mask = numeric > 0
            else:
                mask = values.notna()
            signal_times = df.loc[mask & df["time"].notna(), "time"]
            if not signal_times.empty:
                first_signal = float(signal_times.min())

        return SeriesStats(rows=len(df), non_null=int(values.notna().sum()), mean=mean, maximum=maximum, first_signal=first_signal)

class RicuPyricuComparator:
    """Loads ricu CSVs and easyicu concepts and produces coverage comparisons."""

    def __init__(
        self,
        database: str = "miiv",
        data_path: Optional[Path | str] = None,
        easyicu_data_path: Optional[Path | str] = None,
        patients: Optional[Sequence[int]] = None,
        max_patients: Optional[int] = None,
    ) -> None:
        if database not in SUPPORTED_DATABASES:
            raise ValueError(f"Unsupported database '{database}'. Expected one of {SUPPORTED_DATABASES}.")
        self.database = database
        self.data_path = self._resolve_data_path(database, data_path)
        self.easyicu_data_path = self._resolve_r_path(database, easyicu_data_path)
        self.test_patients = self._resolve_patients(patients, max_patients)
        self._fio2_override: Optional[pd.DataFrame] = None
        self._icustay_times: Optional[pd.DataFrame] = None
        dict_path = REPO_ROOT / "ricu" / "inst" / "extdata" / "config" / "concept-dict.json"
        self._loader_kwargs: Dict[str, object] = {}
        if dict_path.exists():
            self._loader_kwargs["dict_path"] = str(dict_path)

        self._interval_cap_hours = 24 * 365  # prevent runaway window expansion
        patient_goal = (
            len(self.test_patients)
            if isinstance(self.test_patients, list)
            else (max_patients if max_patients and max_patients > 0 else None)
        )
        self._loader_defaults = resolve_loader_defaults(
            patient_goal,
            env_keys=ALIGNMENT_ENV_KEYS,
        )
        loader_kwargs = self._loader_defaults.as_loader_kwargs(progress=False)
        for key, value in loader_kwargs.items():
            if value is None:
                continue
            self._loader_kwargs[key] = value

        self._concept_cache: Dict[Tuple[str, bool], pd.DataFrame] = {}

    def run(self, module_filter: Optional[List[str]] = None) -> Dict[str, FeatureComparison]:
        self._ensure_fio2_patch()
        results: Dict[str, FeatureComparison] = {}
        print(
            f"⚙️  Loader profile [{self.database}]: {self._loader_defaults.summary()}"
        )
        modules_to_test = MODULES
        if module_filter:
            modules_to_test = [m for m in MODULES if m.name in module_filter]
            if not modules_to_test:
                available = ', '.join(m.name for m in MODULES)
                raise ValueError(f"No valid modules found in {module_filter}. Available: {available}")
        for module in modules_to_test:
            comparison = self._compare_module(module)
            results[module.name] = comparison
            self._print_module_summary(module, comparison)
        self._print_dependency_report(results)
        return results

    def _compare_module(self, module: FeatureModule) -> FeatureComparison:
        ricu_series = self._load_r_series(module)
        reference_grid = self._build_time_grid(ricu_series, module)
        easyicu_series = self._load_easyicu_series(module, reference_grid, ricu_series)
        return FeatureComparison(module, ricu_series, easyicu_series)

    def _load_r_series(self, module: FeatureModule) -> Dict[str, pd.DataFrame]:
        file_name = module.ricu_file.format(db=self.database)
        file_path = self.easyicu_data_path / file_name
        if not file_path.exists():
            logger.warning(f"Missing ricu file: {file_path}")
            return {}
        
        # 🚀 OPTIMIZATION: Load only header first to detect columns, then filter during read
        sample_df = pd.read_csv(file_path, nrows=0, low_memory=False)
        id_column = module.id_column if module.id_column in sample_df.columns else self._detect_column(sample_df, module.id_column, ID_CANDIDATES)
        time_column = None
        if module.time_column:
            if module.time_column in sample_df.columns:
                time_column = module.time_column
            else:
                time_column = self._detect_column(sample_df, module.time_column, TIME_CANDIDATES)
        
        # 🚀 OPTIMIZATION: Filter patients during CSV parsing using chunked reader
        if id_column and self.test_patients:
            chunks = []
            chunksize = 50000  # Read in chunks to save memory
            for chunk in pd.read_csv(file_path, low_memory=False, chunksize=chunksize):
                filtered = chunk[chunk[id_column].isin(self.test_patients)]
                if not filtered.empty:
                    chunks.append(filtered)
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=sample_df.columns)
        else:
            df = pd.read_csv(file_path, low_memory=False)
        
        return self._split_wide_table(df, module, detected_id_column=id_column, detected_time_column=time_column)

    def _load_easyicu_series(
        self,
        module: FeatureModule,
        reference_grid: Optional[pd.DataFrame] = None,
        reference_series: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, pd.DataFrame]:
        frames: Dict[str, pd.DataFrame] = {}
        
        # 🔧 关键修复：id_type定义移到try块外,避免UnboundLocalError
        id_type_map = {
            'miiv': 'stay_id',
            'mimic': 'icustay_id',
            'eicu': 'patientunitstayid',
            'aumc': 'admissionid',
            'hirid': 'patientid',
        }
        id_type = id_type_map.get(self.database, 'icustay')
        
        # 🔧 强制模仿 ricu.R：一次性加载整个模块的所有概念
        # ricu.R 使用 load_concepts(c('concept1', 'concept2', ...), interval=hours(1))
        # 这会将所有概念 merge 到共同的时间网格上，缺失值填充为 NA
        try:
            # 🔧 关键修复：明确指定使用 stay_id（或相应数据库的 ID 列）
            # 避免将 stay_id 误认为 subject_id 而加载多个 ICU stay
            
            # 将 patient_ids 包装为字典格式以明确指定 ID 类型
            patient_ids_dict = None
            if self.test_patients:
                patient_ids_dict = {id_type: self.test_patients}
            
            merged_frame = load_concepts(
                list(module.concepts),
                database=self.database,
                patient_ids=patient_ids_dict,
                data_path=str(self.data_path),
                interval="1h" if module.time_column else None,
                merge=True,  # 关键：merge=True 会执行 outer join
                verbose=False,
                _allow_missing_concept=True,
                **self._loader_kwargs,
            )
            
            # 将 merged DataFrame 拆分成单个概念的 series
            if merged_frame is not None and not merged_frame.empty:
                # 🔧 关键修复：批量加载（outer join）会导致 stay_id 列有大量 NA
                # 需要先用 subject_id 填充 stay_id，再进行过滤
                if self.test_patients and 'stay_id' in merged_frame.columns:
                    # 先填充缺失的 stay_id（使用 subject_id 映射）
                    if merged_frame['stay_id'].isna().any() and 'subject_id' in merged_frame.columns:
                        lookup = self._load_icustay_times()
                        if not lookup.empty and 'subject_id' in lookup.columns and 'stay_id' in lookup.columns:
                            # 只映射我们关心的 stay_id
                            target_subjects = lookup[lookup['stay_id'].isin(self.test_patients)]['subject_id'].unique()
                            subject_to_stay = lookup[lookup['subject_id'].isin(target_subjects)][['subject_id', 'stay_id']].drop_duplicates()
                            # 只填充NA的行
                            na_mask = merged_frame['stay_id'].isna()
                            if na_mask.any():
                                merged_frame = merged_frame.merge(
                                    subject_to_stay.rename(columns={'stay_id': 'stay_id_filled'}),
                                    on='subject_id',
                                    how='left'
                                )
                                merged_frame.loc[na_mask, 'stay_id'] = merged_frame.loc[na_mask, 'stay_id_filled']
                                merged_frame = merged_frame.drop(columns=['stay_id_filled'], errors='ignore')
                    
                    # 现在进行过滤
                    original_rows = len(merged_frame)
                    merged_frame = merged_frame[merged_frame['stay_id'].isin(self.test_patients)]
                    filtered_rows = len(merged_frame)
                    # 只在过滤掉大量行时输出
                    if original_rows != filtered_rows and (original_rows - filtered_rows > 100 or filtered_rows < original_rows * 0.5):
                        print(f"   🔍 过滤额外的 stay_id: {original_rows} → {filtered_rows} 行")
                
                for name in module.concepts:
                    if name in merged_frame.columns:
                        # 关键修复：提取概念时，清除其他概念的辅助列（endtime, duration）
                        concept_frame = self._extract_concept_from_merged(merged_frame, name, module)
                        if concept_frame is not None and not concept_frame.empty:
                            series = self._normalize_concept_frame(concept_frame, module, name)
                            if series is not None:
                                frames[name] = series
        except Exception:
            # 批量加载失败是正常的（某些模块的概念无法批量合并），静默回退到单个加载
            # print(f"   ⚠️  批量加载模块 {module.name} 失败，回退到单个加载: {exc}")
            # 回退到原来的单个加载逻辑
            # 🔧 关键修复：单个加载时也需要正确的 patient_ids 格式和过滤
            patient_ids_dict = None
            if self.test_patients:
                patient_ids_dict = {id_type: self.test_patients}
            
            for name in module.concepts:
                try:
                    frame = self._load_concept_frame(name, patient_ids_dict, module)
                    
                    # 🔧 关键修复：过滤掉额外的 stay_id（处理静态概念）
                    if self.test_patients and not frame.empty:
                        # 如果有 stay_id 列，直接过滤
                        if 'stay_id' in frame.columns:
                            original_rows = len(frame)
                            frame = frame[frame['stay_id'].isin(self.test_patients)]
                            filtered_rows = len(frame)
                            # 只在过滤掉大量行时输出（超过100行或超过50%）
                            if original_rows != filtered_rows and (original_rows - filtered_rows > 100 or filtered_rows < original_rows * 0.5):
                                print(f"   🔍 [{name}] 过滤额外的 stay_id: {original_rows} → {filtered_rows} 行")
                        # 如果只有 subject_id，需要转换为 stay_id 再过滤
                        elif 'subject_id' in frame.columns and 'stay_id' not in frame.columns:
                            lookup = self._load_icustay_times()
                            if not lookup.empty and 'subject_id' in lookup.columns and 'stay_id' in lookup.columns:
                                original_rows = len(frame)
                                # 先合并 stay_id
                                frame = frame.merge(
                                    lookup[['subject_id', 'stay_id']].drop_duplicates(),
                                    on='subject_id',
                                    how='left'
                                )
                                # 再过滤
                                frame = frame[frame['stay_id'].isin(self.test_patients)]
                                filtered_rows = len(frame)
                                # 只在过滤掉大量行时输出
                                if original_rows != filtered_rows and (original_rows - filtered_rows > 100 or filtered_rows < original_rows * 0.5):
                                    logger.debug(f"[{name}] 通过 subject_id→stay_id 过滤: {original_rows} → {filtered_rows} 行")
                    
                    if name == "fio2" and self._fio2_override is not None:
                        frame = self._fio2_override.copy()
                    series = self._normalize_concept_frame(frame, module, name)
                    if self._should_retry_per_patient(series, frame):
                        retry = self._reload_concept_per_patient(name, module)
                        if retry is not None:
                            series = retry
                    if series is not None:
                        frames[name] = series
                
                except Exception as exc:
                    logger.warning(f"concept {name} failed in module {module.name}: {exc}")
                    continue

        # 批量加载后不需要再次对齐，因为已经在共同的时间网格上了
        align_grid: Optional[pd.DataFrame] = None
        if module.time_column:
            if reference_grid is not None and not reference_grid.empty:
                align_grid = reference_grid.copy()
            elif frames:
                align_grid = self._build_time_grid(frames, module)

        if align_grid is not None and not align_grid.empty:
            grid = align_grid.copy()
            grid["id"] = pd.to_numeric(grid["id"], errors="coerce")
            grid["time"] = pd.to_numeric(grid["time"], errors="coerce")
            grid = grid.dropna(subset=["id", "time"]).drop_duplicates().reset_index(drop=True)
            if not grid.empty:
                for name, df in list(frames.items()):
                    if "time" not in df.columns:
                        continue
                    df = df.copy()
                    df["id"] = pd.to_numeric(df["id"], errors="coerce")
                    df["time"] = pd.to_numeric(df["time"], errors="coerce")
                    df = df.dropna(subset=["id", "time"])
                    aligned = grid.merge(df, on=["id", "time"], how="left")
                    
                    # 静态概念填充逻辑：
                    # 静态概念（los_icu, age, sex等，即target=id_tbl）应该在所有时间点填充相同的值
                    # 时间序列概念（bnd, abx, death等）只保留实际测量点
                    # 注意：death是lgl_cncpt，只在死亡时刻有TRUE，其他时间点应该是NA
                    static_concepts = {"los_icu", "age", "sex", "bmi", "height", "weight"}
                    is_static_concept = name in static_concepts
                    
                    if "value" in aligned.columns and "id" in aligned.columns:
                        for patient_id in aligned["id"].unique():
                            if pd.isna(patient_id):
                                continue
                            patient_mask = aligned["id"] == patient_id
                            patient_values = aligned.loc[patient_mask, "value"]
                            non_na_values = patient_values.dropna()
                            
                            # 静态概念：有值且所有值相同 → forward-fill
                            if is_static_concept and len(non_na_values) > 0 and non_na_values.nunique() == 1:
                                fill_value = non_na_values.iloc[0]
                                aligned.loc[patient_mask, "value"] = fill_value
                    
                    # 清理额外的ID列（subject_id, hadm_id等），只保留id, time, value
                    keep_cols = ["id", "time"]
                    if "value" in aligned.columns:
                        keep_cols.append("value")
                    # 保留必要的辅助列（如endtime, duration）
                    for aux_col in ["endtime", "duration"]:
                        if aux_col in aligned.columns:
                            keep_cols.append(aux_col)
                    aligned = aligned[keep_cols]
                    
                    frames[name] = aligned
                align_grid = grid
            else:
                align_grid = None

        base_placeholder = align_grid.copy() if module.time_column and align_grid is not None and not align_grid.empty else None

        for concept in module.concepts:
            if concept in frames:
                continue
            placeholder: Optional[pd.DataFrame] = None
            # 检查reference_series中该概念是否有时间列
            # 静态概念（如los_icu, death）不应使用时间网格placeholder
            is_time_series_concept = False
            if reference_series and concept in reference_series:
                ref_df = reference_series[concept]
                if not ref_df.empty and "time" in ref_df.columns:
                    is_time_series_concept = True
            
            if base_placeholder is not None and is_time_series_concept:
                # 只为时间序列概念使用时间网格placeholder
                placeholder = base_placeholder.assign(value=np.nan)
            elif reference_series and concept in reference_series:
                ref_df = reference_series[concept]
                if not ref_df.empty and "id" in ref_df.columns:
                    cols = ["id"]
                    if "time" in ref_df.columns:
                        cols.append("time")
                    placeholder = ref_df[cols].copy()
                    placeholder["value"] = np.nan
            if placeholder is not None:
                frames[concept] = placeholder.reset_index(drop=True)

        return frames

    def _split_wide_table(
        self,
        df: pd.DataFrame,
        module: FeatureModule,
        detected_id_column: Optional[str] = None,
        detected_time_column: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        id_column = detected_id_column
        if not id_column or id_column not in df.columns:
            id_column = self._detect_column(df, module.id_column, ID_CANDIDATES)
        if id_column is None or id_column not in df.columns:
            return result
        time_column = detected_time_column
        if module.time_column:
            if not time_column or time_column not in df.columns:
                time_column = self._detect_column(df, module.time_column, TIME_CANDIDATES)
        base_cols = [id_column]
        if time_column and time_column in df.columns:
            base_cols.append(time_column)
        for column in df.columns:
            if column in base_cols:
                continue
            subset_cols = base_cols + [column]
            series = df[subset_cols].copy()
            rename_map = {column: "value", id_column: "id"}
            if time_column and time_column in series.columns:
                rename_map[time_column] = "time"
            series = series.rename(columns=rename_map)
            if "time" in series.columns and not pd.api.types.is_numeric_dtype(series["time"]):
                id_ref = series["id"] if "id" in series.columns else None
                series["time"] = self._time_to_hours(series["time"], id_ref)
            series = series.dropna(subset=["id"])
            result[column] = series.reset_index(drop=True)
        return result

    def _normalize_concept_frame(
        self,
        frame: pd.DataFrame,
        module: FeatureModule,
        concept: str,
        *,
        force_id: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        if frame is None or (hasattr(frame, "empty") and frame.empty):
            return None
        if isinstance(frame, pd.Series):
            df = frame.to_frame(name="value").reset_index()
        else:
            df = frame.copy()
        if df.empty:
            return None
        
        # Enhanced ID column detection: try preferred column, then fallbacks
        id_col = self._detect_column(df, module.id_column, ID_CANDIDATES)
        
        # Special handling: if we got subject_id but need stay_id (or vice versa),
        # convert it now using icustays mapping BEFORE renaming to "id"
        needs_conversion = False
        if id_col == "subject_id" and module.id_column == "stay_id":
            needs_conversion = True
        elif id_col == "stay_id" and module.id_column == "subject_id":
            needs_conversion = True
        
        # Convert BEFORE renaming to preserve original column name
        if needs_conversion:
            df = self._convert_id_column(df, from_col=id_col, to_col=module.id_column)
            # Update id_col to reflect the converted column
            id_col = module.id_column
        
        if id_col is None:
            return None
        rename_map = {id_col: "id"}
        cols = [id_col]
        time_col = None
        if module.time_column:
            time_col = self._detect_column(df, module.time_column, TIME_CANDIDATES)
            if time_col:
                rename_map[time_col] = "time"
                cols.append(time_col)
            elif module.time_column:
                # 对于需要时间列的module，如果概念没有时间列（如death, los_icu），
                # 添加time=0作为默认时间点，以便后续对齐时可以扩展到整个时间网格
                df["time"] = 0.0
                rename_map["time"] = "time"
                cols.append("time")
                time_col = "time"
        value_col = concept if concept in df.columns else self._detect_value_column(df, cols)
        if value_col is None:
            return None
        rename_map[value_col] = "value"
        cols.append(value_col)

        # keep auxiliary columns so we can expand windows or repair ids later
        end_col = self._detect_column(df, "endtime", ["endtime", "end_time", "stop", f"{concept}_end", f"{concept}_endtime"])
        duration_col = self._detect_column(df, f"{concept}_dur", [f"{concept}_dur", f"{concept}_duration", "duration", "dur", "hours", "length"])
        backup_ids = [candidate for candidate in ("subject_id", "hadm_id", "patientunitstayid") if candidate in df.columns]
        for extra in filter(None, [end_col, duration_col]):
            if extra not in cols and extra in df.columns:
                cols.append(extra)
        for backup in backup_ids:
            if backup not in cols:
                cols.append(backup)
                rename_map.setdefault(backup, backup)
        if end_col:
            rename_map[end_col] = "endtime"
        if duration_col:
            rename_map[duration_col] = "duration"
        df = df[cols].rename(columns=rename_map)
        if "value" in df.columns and pd.api.types.is_timedelta64_dtype(df["value"]):
            df["value"] = df["value"].dt.total_seconds() / 3600.0
        # Drop duplicate columns (e.g., repeated component outputs) while keeping first values
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        if "time" in df.columns and not pd.api.types.is_numeric_dtype(df["time"]):
            id_ref = df["id"] if "id" in df.columns else None
            df["time"] = self._time_to_hours(df["time"], id_ref)
        if "time" in df.columns:
            df = df.dropna(subset=["time"])
        
        # Fill any remaining missing IDs
        df = self._fill_missing_ids(df)
        
        if force_id is not None and "id" in df.columns:
            df["id"] = df["id"].fillna(force_id)
        df = df.dropna(subset=["id"]).reset_index(drop=True)
        df = self._expand_interval_rows(df, module, concept)
        return df

    def _extract_concept_from_merged(
        self,
        merged_frame: pd.DataFrame,
        concept: str,
        module: FeatureModule,
    ) -> pd.DataFrame:
        """从批量加载的 merged DataFrame 中提取单个概念，只保留该概念相关的列。
        
        关键：保留共享的辅助列（endtime, duration, stoptime），以便后续展开窗口。
        """
        if merged_frame is None or merged_frame.empty:
            return pd.DataFrame()
        
        # 基础列
        id_cols = ["stay_id", "subject_id", "hadm_id"] if module.id_column else []
        time_cols = ["starttime", "charttime"] if module.time_column else []
        base_cols = [col for col in id_cols + time_cols if col in merged_frame.columns]
        
        # 概念值列
        if concept not in merged_frame.columns:
            return pd.DataFrame()
        
        # 概念特定的辅助列（以概念名开头的列）
        concept_aux_cols = [
            col for col in merged_frame.columns
            if col.startswith(f"{concept}_")
        ]
        
        # 共享的辅助列（所有概念共用的时间列）
        shared_aux_cols = [
            col for col in ["endtime", "stoptime", "duration", "dose_unit_rx"]
            if col in merged_frame.columns
        ]
        
        # 选择列：基础列 + 概念值 + 概念特定辅助列 + 共享辅助列
        keep_cols = list(set(base_cols + [concept] + concept_aux_cols + shared_aux_cols))
        keep_cols = [col for col in keep_cols if col in merged_frame.columns]
        
        result = merged_frame[keep_cols].copy()
        
        # 🔧 关键修复：批量加载时，endtime/duration 是共享列，可能不属于当前概念
        # 判断当前概念是否需要这些时间列：
        # - *_rate, *_dur, mech_vent 等窗口型概念：需要 endtime/duration
        # - 其他点值概念（norepi_equiv, vaso_ind等）：不需要，应移除
        concept_lower = concept.lower()
        needs_window = (
            concept_lower.endswith('_rate') or 
            concept_lower.endswith('_dur') or
            concept_lower in {'mech_vent', 'vent_ind'}
        )
        
        if not needs_window:
            # 点值概念：完全移除 endtime/duration，避免错误展开
            for time_col in ["endtime", "stoptime", "duration"]:
                if time_col in result.columns:
                    result = result.drop(columns=[time_col])
        else:
            # 窗口型概念：清除没有值的行的时间信息，并转换 endtime 为小时数
            if concept in result.columns:
                concept_has_value = result[concept].notna()
                for time_col in ["endtime", "stoptime", "duration"]:
                    if time_col in result.columns:
                        # 修复 dtype 兼容性：先转换列类型再赋值
                        if time_col != "duration":
                            if not pd.api.types.is_datetime64_any_dtype(result[time_col]):
                                result[time_col] = pd.to_datetime(result[time_col], errors='coerce')
                            result.loc[~concept_has_value, time_col] = pd.NaT
                        else:
                            result.loc[~concept_has_value, time_col] = None
            
            # 🔧 关键修复：转换 endtime 从 datetime64 到小时数
            # 问题：批量加载时 starttime 已是小时数，但 endtime 仍是时间戳
            # 需要用每行对应的 stay_id 的 intime 来转换
            if "endtime" in result.columns and not pd.api.types.is_numeric_dtype(result["endtime"]):
                # 加载 icustay_times 获取 intime
                icu = self._load_icustay_times()
                if not icu.empty and "intime" in icu.columns:
                    # 确定 ID 列名（优先 stay_id）
                    id_col = None
                    for candidate in ["stay_id", "subject_id", "hadm_id"]:
                        if candidate in result.columns:
                            id_col = candidate
                            # 必须确保 icu 表中也有这个列
                            if id_col == "stay_id" and "stay_id" in icu.columns:
                                break
                            elif id_col == "subject_id" and "subject_id" in icu.columns:
                                break
                            elif id_col == "hadm_id" and "hadm_id" in icu.columns:
                                break
                            id_col = None
                    
                    if id_col and id_col in icu.columns:
                        # 构建 intime 映射表，只保留需要的 ID
                        result_ids = result[id_col].unique()
                        icu_map = icu[icu[id_col].isin(result_ids)][[id_col, "intime"]].dropna().drop_duplicates()
                        icu_map["intime"] = pd.to_datetime(icu_map["intime"], errors="coerce")
                        if hasattr(icu_map["intime"].dtype, 'tz') and icu_map["intime"].dt.tz is not None:
                            icu_map["intime"] = icu_map["intime"].dt.tz_localize(None)
                        
                        # 合并获取 intime
                        result = result.merge(icu_map, on=id_col, how="left")
                        
                        # 转换 endtime
                        end_dt = pd.to_datetime(result["endtime"], errors="coerce")
                        if hasattr(end_dt.dtype, 'tz') and end_dt.dt.tz is not None:
                            end_dt = end_dt.dt.tz_localize(None)
                        
                        result["endtime"] = (end_dt - result["intime"]).dt.total_seconds() / 3600.0
                        result = result.drop(columns=["intime"], errors="ignore")
                        
                        # 清除异常值（可能由于 ID 匹配失败导致）
                        if pd.api.types.is_numeric_dtype(result["endtime"]):
                            # 先转换为object类型以避免FutureWarning
                            mask = (result["endtime"] > 10000) | (result["endtime"] < -10000)
                            if mask.any():
                                result["endtime"] = result["endtime"].astype(object)
                                result.loc[mask, "endtime"] = pd.NaT
        
        # 注意：不过滤空值行，保持 ricu 的完整时间网格
        
        return result
    
    def _load_concept_frame(
        self,
        concept: str,
        patient_ids: Optional[Sequence[int]],
        module: FeatureModule,
    ) -> pd.DataFrame:
        cache_key = (concept, bool(module.time_column))
        cached = self._concept_cache.get(cache_key)
        if cached is not None:
            return cached

        frame = load_concepts(
            concept,
            database=self.database,  # 修正：使用 database= 而不是 src=
            patient_ids=patient_ids,
            data_path=str(self.data_path),
            interval="1h" if module.time_column else None,
            merge=True,
            verbose=False,
            _allow_missing_concept=True,
            **self._loader_kwargs,
        )
        self._concept_cache[cache_key] = frame
        return frame

    def _should_retry_per_patient(
        self,
        series: Optional[pd.DataFrame],
        raw_frame: Optional[pd.DataFrame],
    ) -> bool:
        if not self.test_patients:
            return False
        if raw_frame is None or not isinstance(raw_frame, pd.DataFrame) or raw_frame.empty:
            return False
        if series is None or series.empty:
            return True
        if "id" not in series.columns:
            return True
        return not series["id"].notna().any()

    def _reload_concept_per_patient(self, concept: str, module: FeatureModule) -> Optional[pd.DataFrame]:
        if not self.test_patients:
            return None
        per_patient: List[pd.DataFrame] = []
        for stay_id in self.test_patients:
            try:
                frame = self._load_concept_frame(concept, [stay_id], module)
            except Exception as exc:
                logger.warning("Failed reloading %s for stay %s: %s", concept, stay_id, exc)
                continue
            normalized = self._normalize_concept_frame(frame, module, concept, force_id=stay_id)
            if normalized is not None and not normalized.empty:
                per_patient.append(normalized)
        if not per_patient:
            return None
        combined = pd.concat(per_patient, ignore_index=True)
        sort_cols = ["id"] + (["time"] if "time" in combined.columns else [])
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
        return combined

    def _convert_id_column(self, df: pd.DataFrame, from_col: str, to_col: str) -> pd.DataFrame:
        """Convert between ID column types (e.g., subject_id <-> stay_id).
        
        This function expects the ORIGINAL column name (e.g., 'subject_id')
        and replaces it with the target column (e.g., 'stay_id').
        """
        if from_col not in df.columns:
            logger.warning(f"Cannot convert {from_col} to {to_col}: {from_col} not in columns")
            return df
        
        lookup = self._load_icustay_times()
        if lookup.empty or from_col not in lookup.columns or to_col not in lookup.columns:
            logger.warning(f"Cannot convert {from_col} to {to_col}: missing in lookup table")
            return df
        
        # Build mapping table: from_col → to_col
        bridge = lookup[[from_col, to_col]].dropna().drop_duplicates()
        if bridge.empty:
            logger.warning(f"Cannot convert {from_col} to {to_col}: no valid mappings")
            return df
        
        # Merge directly on original column name
        merged = df.merge(bridge, on=from_col, how="left")
        
        # Drop the old column, keep the new one
        if from_col in merged.columns:
            merged = merged.drop(columns=[from_col])
        
        return merged
    
    def _fill_missing_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing ID values using backup ID columns."""
        if "id" not in df.columns:
            return df
        
        # If all IDs are present, nothing to do
        if df["id"].notna().all():
            return df
        
        lookup = self._load_icustay_times()
        if lookup.empty:
            return df

        merged = df.copy()
        for column in ("hadm_id", "subject_id"):
            if column not in merged.columns or column not in lookup.columns:
                continue
            bridge = lookup[["stay_id", column]].dropna().drop_duplicates()
            if bridge.empty:
                continue
            merged = merged.merge(bridge, on=column, how="left", suffixes=("", "_mapped"))
            mapped_col = "stay_id"
            if mapped_col in merged.columns:
                merged["id"] = merged["id"].fillna(merged[mapped_col])
                merged = merged.drop(columns=[mapped_col])
        
        # Clean up backup ID columns
        for cleanup in ("subject_id", "hadm_id", "patientunitstayid"):
            if cleanup in merged.columns:
                merged = merged.drop(columns=[cleanup])
        return merged

    def _expand_interval_rows(self, df: pd.DataFrame, module: FeatureModule, concept: str) -> pd.DataFrame:
        # Check if we have time column (required for expansion)
        if "time" not in df.columns:
            return df.drop(columns=["endtime", "duration"], errors="ignore")

        has_end = "endtime" in df.columns and df["endtime"].notna().any()
        has_duration = "duration" in df.columns and df["duration"].notna().any()

        # Duration-style concepts (e.g., *_dur) already encode exposure length and
        # should not be expanded into per-hour series.
        concept_lower = concept.lower()
        if concept_lower.endswith("_dur"):
            return df.drop(columns=["endtime", "duration"], errors="ignore")
        
        # Logical/boolean concepts (abx, samp, etc.) are event indicators and
        # should NOT be expanded into continuous time series. They represent
        # discrete events (e.g., antibiotic administration started at time X),
        # not continuous states. Expanding them would incorrectly fill gaps.
        logical_concepts = {
            "abx", "samp", "cort", "vaso_ind", "dobu60", 
            "susp_inf", "sep3", "ett_gcs", "avpu"
        }
        if concept_lower in logical_concepts:
            return df.drop(columns=["endtime", "duration"], errors="ignore")

        # Do not fabricate synthetic durations when the upstream concept already
        # provides discrete time points (ricu treats those as point events).  The
        # previous heuristic expanded every measurement until the next one,
        # drastically overestimating vasopressor exposure compared to ricu.
        if not has_end and not has_duration:
            return df.drop(columns=["endtime", "duration"], errors="ignore")
        
        # Expand window-based concepts (mech_vent, etc.) from all modules, not just med
        # This includes respiratory concepts like mech_vent that have start/end times

        working = df.copy()
        
        # 🔧 关键修复：批量加载时，endtime 是共享列，可能包含其他概念的值
        # 只展开当前概念有值的行，忽略只有 endtime 但没有 value 的行
        if "value" in working.columns and (has_end or has_duration):
            # 只保留有值的行
            has_value = working["value"].notna()
            working = working[has_value].copy()
            
            # 如果过滤后没有行了，直接返回空
            if working.empty:
                return pd.DataFrame(columns=["id", "time", "value"])

        if has_end and not pd.api.types.is_numeric_dtype(working["endtime"]):
            icu = self._load_icustay_times()
            
            if not icu.empty and "intime" in icu.columns:
                # Determine the ID column in working DataFrame
                id_col_in_working = "id" if "id" in working.columns else "stay_id" if "stay_id" in working.columns else None
                
                if id_col_in_working:
                    icu_map = icu[["stay_id", "intime"]].dropna().drop_duplicates()
                    # 统一处理时区：移除时区信息以避免比较错误
                    icu_map["intime"] = pd.to_datetime(icu_map["intime"], errors="coerce")
                    if hasattr(icu_map["intime"].dtype, 'tz') and icu_map["intime"].dt.tz is not None:
                        icu_map["intime"] = icu_map["intime"].dt.tz_localize(None)
                    
                    # Rename to match working's ID column
                    if id_col_in_working != "stay_id":
                        icu_map = icu_map.rename(columns={"stay_id": id_col_in_working})
                    
                    working = working.merge(icu_map, on=id_col_in_working, how="left")
                    
                    # 统一 endtime 时区处理
                    end_dt = pd.to_datetime(working["endtime"], errors="coerce")
                    if hasattr(end_dt.dtype, 'tz') and end_dt.dt.tz is not None:
                        end_dt = end_dt.dt.tz_localize(None)
                    
                    working["endtime"] = (
                        (end_dt - working["intime"]).dt.total_seconds() / 3600.0
                    )
                    working = working.drop(columns=["intime"], errors="ignore")
                else:
                    # Fallback: try to parse as timestamp
                    working["endtime"] = self._time_to_hours(working["endtime"], None)
            else:
                working["endtime"] = self._time_to_hours(working["endtime"], working.get("id"))
        if has_duration:
            if pd.api.types.is_timedelta64_dtype(working["duration"]):
                working["duration"] = working["duration"].dt.total_seconds() / 3600.0
            working["duration"] = pd.to_numeric(working["duration"], errors="coerce")

        starts = pd.to_numeric(working["time"], errors="coerce")
        if has_end:
            ends = pd.to_numeric(working["endtime"], errors="coerce")
        elif has_duration:
            ends = starts + working["duration"].fillna(0)
        else:
            ends = starts

        records: List[dict] = []
        skipped = 0
        for idx, (start, end, value, stay_id) in enumerate(zip(starts, ends, working.get("value"), working.get("id"))):
            if pd.isna(start) or pd.isna(end) or pd.isna(stay_id) or pd.isna(value):
                continue
            if end < start:
                end = start
            span = min(end - start, self._interval_cap_hours)
            if span <= 0:
                records.append({"id": stay_id, "time": float(start), "value": value})
                continue
            if span >= self._interval_cap_hours:
                skipped += 1
            # 🔧 修复：与 ricu 的 expand() 行为一致
            # ricu 使用 seq(min, max, step_size)，不会超过 max 值
            # 例如：start=28.93, end=29.23 → 只生成 hour 28, 29（因为 29 <= 29.23）
            # 而非之前的 ceil(29.23)=30 → hour 28, 29, 30
            start_hour = int(math.floor(start))
            end_hour = int(math.floor(min(end, start + self._interval_cap_hours)))
            for hour in range(start_hour, end_hour + 1):
                records.append({"id": stay_id, "time": float(hour), "value": value})
        if not records:
            return df.drop(columns=["endtime", "duration"], errors="ignore")
        expanded = pd.DataFrame.from_records(records)
        # 🔧 修复：按 (id, time) 聚合，保留最后一个值
        # 这与 ricu 的 aggregate 行为一致：同一时间点有多个测量时取最后一个
        # 之前使用 drop_duplicates(subset=["id", "time", "value"]) 不会去除
        # 同一 (id, time) 但 value 不同的行，导致 merge 时产生重复
        expanded = expanded.groupby(["id", "time"], as_index=False).last()
        return expanded

    def _detect_column(self, df: pd.DataFrame, preferred: Optional[str], fallbacks: Iterable[str]) -> Optional[str]:
        if preferred and preferred in df.columns:
            return preferred
        for candidate in fallbacks:
            if candidate in df.columns:
                return candidate
        return None

    def _detect_value_column(self, df: pd.DataFrame, exclude: Iterable[str]) -> Optional[str]:
        exclude_set = set(exclude)
        numeric_cols = [c for c in df.columns if c not in exclude_set]
        return numeric_cols[0] if numeric_cols else None

    def _time_to_hours(self, series: pd.Series, id_series: Optional[pd.Series]) -> pd.Series:
        if series.empty:
            return series
        if pd.api.types.is_datetime64_any_dtype(series):
            clean = series.dt.tz_localize(None)
            if id_series is not None:
                return clean.groupby(id_series).transform(lambda s: (s - s.min()).dt.total_seconds() / 3600.0)
            return (clean - clean.min()).dt.total_seconds() / 3600.0
        if pd.api.types.is_timedelta64_dtype(series):
            return series.dt.total_seconds() / 3600.0
        return pd.to_numeric(series, errors="coerce")

    def _build_time_grid(
        self,
        series_dict: Dict[str, pd.DataFrame],
        module: FeatureModule,
    ) -> Optional[pd.DataFrame]:
        if not module.time_column:
            return None
        frames = [
            df[["id", "time"]]
            for df in series_dict.values()
            if isinstance(df, pd.DataFrame) and {"id", "time"}.issubset(df.columns)
        ]
        if not frames:
            return None
        grid = (
            pd.concat(frames, ignore_index=True)
            .dropna(subset=["id", "time"])
            .drop_duplicates()
            .sort_values(["id", "time"])
            .reset_index(drop=True)
        )
        return grid if not grid.empty else None

    def _ensure_fio2_patch(self) -> None:
        """Apply fio2 patch only for MIMIC-IV database."""
        if self._fio2_override is not None:
            return
        # Only apply this patch for miiv database (chartevents.parquet is miiv-specific)
        if self.database not in ("miiv", "mimiciv"):
            return
        base = self._load_easyicu_fio2()
        base_has_values = (
            isinstance(base, pd.DataFrame)
            and not base.empty
            and "fio2" in base.columns
            and base["fio2"].notna().any()
        )
        if base_has_values:
            # 现在easyicu的fio2加载已经可以解析百分号字符串，直接使用标准结果即可
            return
        raw = self._load_raw_fio2()
        frames = [df for df in (base, raw) if isinstance(df, pd.DataFrame) and not df.empty]
        if not frames:
            return
        merged = pd.concat(frames, ignore_index=True)
        # Check if stay_id column exists before using it
        if "stay_id" not in merged.columns:
            return
        merged = merged.dropna(subset=["stay_id"])
        if "charttime" not in merged.columns:
            return
        merged["charttime"] = pd.to_numeric(merged["charttime"], errors="coerce")
        merged["charttime"] = np.floor(merged["charttime"].astype(float))
        merged = merged.sort_values(["stay_id", "charttime"])
        merged = merged.drop_duplicates(subset=["stay_id", "charttime"], keep="first")
        self._fio2_override = merged.reset_index(drop=True)

    def _load_easyicu_fio2(self) -> pd.DataFrame:
        try:
            # 🔧 关键修复：明确指定使用 stay_id，避免歧义
            id_type_map = {
                'miiv': 'stay_id',
                'mimic': 'icustay_id',
                'eicu': 'patientunitstayid',
                'aumc': 'admissionid',
                'hirid': 'patientid',
            }
            id_type = id_type_map.get(self.database, 'stay_id')
            patient_ids_dict = None
            if self.test_patients:
                patient_ids_dict = {id_type: self.test_patients}

            df = load_concepts(
                "fio2",
                patient_ids=patient_ids_dict,
                database=self.database,
                data_path=str(self.data_path),
                merge=True,
                verbose=False,
                **self._loader_kwargs,
            )
        except Exception:
            return pd.DataFrame(columns=["stay_id", "charttime", "fio2"])
        keep = [c for c in ["stay_id", "charttime", "fio2"] if c in df.columns]
        return df[keep].copy() if keep else pd.DataFrame(columns=["stay_id", "charttime", "fio2"])

    def _load_raw_fio2(self) -> pd.DataFrame:
        chart_path = Path(self.data_path) / "chartevents.parquet"
        if not chart_path.exists():
            return pd.DataFrame(columns=["stay_id", "charttime", "fio2"])
        
        # 🚀 OPTIMIZATION: Use pushdown filters if possible
        columns = ["stay_id", "charttime", "itemid", "valuenum"]
        filters = None
        if self.test_patients:
            # PyArrow filters format: [('col', 'op', val), ...]
            # For 'in' operator, val must be a set/list
            filters = [('stay_id', 'in', set(self.test_patients))]
        
        try:
            chart = pd.read_parquet(chart_path, columns=columns, filters=filters)
        except Exception:
            # Fallback if filters not supported or other error
            chart = pd.read_parquet(chart_path, columns=columns)
            if self.test_patients:
                chart = chart[chart["stay_id"].isin(self.test_patients)]

        chart = chart[chart["itemid"].isin(FIO2_ITEMIDS)]
        chart = chart.dropna(subset=["valuenum", "charttime"])
        if chart.empty:
            return pd.DataFrame(columns=["stay_id", "charttime", "fio2"])
        times = self._load_icustay_times()
        chart = chart.merge(times, on="stay_id", how="left")
        chart = chart.dropna(subset=["intime"])
        chart["charttime"] = (
            pd.to_datetime(chart["charttime"]) - pd.to_datetime(chart["intime"])
        ).dt.total_seconds() / 3600.0
        chart["charttime"] = np.floor(chart["charttime"].astype(float))
        chart = chart.dropna(subset=["charttime"])
        chart["fio2"] = pd.to_numeric(chart["valuenum"], errors="coerce")
        return chart[["stay_id", "charttime", "fio2"]].dropna(subset=["fio2"])

    def _load_icustay_times(self) -> pd.DataFrame:
        if self._icustay_times is not None:
            return self._icustay_times
        icu_path = Path(self.data_path) / "icustays.parquet"
        if icu_path.exists():
            desired = ["stay_id", "subject_id", "hadm_id", "intime", "outtime"]
            try:
                df = pd.read_parquet(icu_path, columns=desired)
            except Exception:
                df = pd.read_parquet(icu_path)
                keep = [col for col in desired if col in df.columns]
                df = df[keep] if keep else df
            for time_col in ("intime", "outtime"):
                if time_col in df.columns:
                    df[time_col] = pd.to_datetime(df[time_col])
        else:
            df = pd.DataFrame(columns=["stay_id", "subject_id", "hadm_id", "intime", "outtime"])
        self._icustay_times = df
        return df

    def _print_module_summary(self, module: FeatureModule, comparison: FeatureComparison) -> None:
        print(f"\n[{module.name.upper()}] {module.description}")
        metrics = comparison.compute_metrics()
        if not metrics:
            print("   ⚠️  No comparable concepts loaded.")
            return
        for name, stats in metrics.items():
            coverage_gap = stats["coverage_gap"]
            status = stats["status"]
            mean_diff = stats["mean_diff"]
            mean_repr = "  N/A" if mean_diff is None else f"{mean_diff:6.2f}"
            print(
                f"   - {name:12s} "
                f"ricu_rows={stats['ricu_rows']:4d} "
                f"py_rows={stats['easyicu_rows']:4d} "
                f"cov_gap={coverage_gap:6.2f} "
                f"mean_diff={mean_repr} "
                f"status={status}"
            )

    def _print_dependency_report(self, results: Dict[str, FeatureComparison]) -> None:
        print("\nSOFA dependency coverage:")
        for component, deps in SOFA_COMPONENT_DEPENDENCIES.items():
            missing = [dep for dep in deps if not self._has_concept(results, dep)]
            if missing:
                print(f" - {component}: missing {', '.join(missing)}")
            else:
                print(f" - {component}: OK")

    def _has_concept(self, results: Dict[str, FeatureComparison], concept: str) -> bool:
        for comparison in results.values():
            series = comparison.easyicu_series.get(concept)
            if series is None or series.empty:
                continue
            if "value" not in series.columns:
                continue
            value_col = series["value"]
            if isinstance(value_col, pd.DataFrame):
                has_values = bool(value_col.stack().notna().any())
            else:
                has_values = bool(pd.Series(value_col).notna().any())
            if has_values:
                return True
        return False

    def _resolve_data_path(self, database: str, override: Optional[Path | str]) -> Path:
        if override is not None:
            path = Path(override).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Data path {path} does not exist")
            return path
        default = DEFAULT_DATA_PATHS.get(database)
        if default is None:
            raise ValueError(f"No default data path for database '{database}'. Please pass --data-path.")
        path = Path(default).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Default data path for '{database}' not found at {path}")
        return path

    def _resolve_r_path(self, database: str, override: Optional[Path | str]) -> Path:
        if override is not None:
            candidate = Path(override).expanduser().resolve()
            if candidate.is_dir():
                return candidate
            base = candidate / database
            if base.is_dir():
                return base
            raise FileNotFoundError(f"Cannot locate ricu data for '{database}' under {candidate}")
        candidate = (DEFAULT_R_DATA_ROOT / database).resolve()
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Cannot locate default ricu data for '{database}' (looked in {candidate})")

    def _resolve_patients(
        self,
        override: Optional[Sequence[int]],
        max_patients: Optional[int],
    ) -> Optional[List[int]]:
        if override:
            ids = [int(pid) for pid in override]
        else:
            ids = list(DEFAULT_PATIENTS.get(self.database, []))
        if not ids:
            ids = self._discover_patient_ids(max_patients or 5)
        if max_patients is not None and ids:
            ids = ids[:max_patients]
        return ids or None

    def _discover_patient_ids(self, limit: int) -> List[int]:
        table, column = PATIENT_ID_SOURCES.get(self.database, (None, None))
        if table is None or column is None:
            return []
        path = Path(self.data_path) / table
        if not path.exists():
            return []
        try:
            df = pd.read_parquet(path, columns=[column])
        except Exception:
            return []
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        unique_ids: List[int] = []
        for value in series:
            candidate = int(value)
            if candidate not in unique_ids:
                unique_ids.append(candidate)
            if len(unique_ids) >= limit:
                break
        return unique_ids

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ricu CSV exports with easyicu outputs.")
    parser.add_argument(
        "-d",
        "--database",
        choices=SUPPORTED_DATABASES,
        nargs="+",
        default=["miiv"],
        help="Database(s) to evaluate (default: miiv).",
    )
    parser.add_argument(
        "--data-path",
        help="Override easyicu data path template. Use {db} to substitute the database name when running multiple targets.",
    )
    parser.add_argument(
        "--ricu-data",
        help="Override ricu CSV path template. Use {db} placeholder for multi-database runs. Defaults to <repo>/ricu_data/<db>.",
    )
    parser.add_argument(
        "-p",
        "--patients",
        nargs="+",
        type=int,
        help="Explicit patient IDs to test. Applies to all databases in this run.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        help="Limit the number of patients pulled from the default/discovered lists.",
    )
    return parser.parse_args(argv)

def _format_path_template(template: Optional[str], database: str, fallback: Optional[Path]) -> Path:
    if not template:
        if fallback is None:
            raise ValueError(f"No default path available for database '{database}'. Please supply --data-path/--ricu-data.")
        return fallback
    value = template.format(db=database)
    return Path(value).expanduser().resolve()

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    for index, database in enumerate(args.database):
        data_default = DEFAULT_DATA_PATHS.get(database)
        data_path = _format_path_template(args.data_path, database, data_default)
        r_data_default = (DEFAULT_R_DATA_ROOT / database).resolve()
        ricu_path = _format_path_template(args.ricu_data, database, r_data_default)
        comparator = RicuPyricuComparator(
            database=database,
            data_path=data_path,
            easyicu_data_path=ricu_path,
            patients=args.patients,
            max_patients=args.max_patients,
        )
        header = f"\n{'=' * 30} Database: {database} {'=' * 30}"
        if index == 0:
            header = header.lstrip("\n")
        print(header)
        comparator.run()

__all__ = [
    "FeatureModule",
    "FeatureComparison",
    "RicuPyricuComparator",
    "MODULES",
    "SOFA_COMPONENT_DEPENDENCIES",
    "main",
]
