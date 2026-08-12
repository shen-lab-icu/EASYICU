"""
患者筛选模块 - 支持根据人口统计学和临床特征筛选ICU患者队列

支持的筛选条件:
- 年龄范围
- 是否首次入ICU
- ICU住院时长
- 性别
- 入院类型

用法示例:
    >>> from easyicu.patient_filter import PatientFilter, filter_patients
    >>> 
    >>> # 快速筛选
    >>> patient_ids = filter_patients(
    ...     database='miiv',
    ...     data_path='/path/to/data',
    ...     age_min=18, age_max=80,
    ...     first_icu_stay=True,
    ...     los_min=24  # 至少24小时
    ... )
    >>> 
    >>> # 使用筛选器对象（可复用）
    >>> pf = PatientFilter(database='miiv', data_path='/path/to/data')
    >>> patients = pf.filter(age_min=18, first_icu_stay=True)
    >>> print(pf.get_filter_summary())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

import pandas as pd

from .databases.profiles import normalize_database_key

logger = logging.getLogger(__name__)


class PatientFilterCriterionError(RuntimeError):
    """A requested cohort criterion could not be applied faithfully."""

    def __init__(self, code: str, criterion: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.criterion = criterion


def _hospital_survival_from_expire_flag(values: pd.Series) -> pd.Series:
    """Map the standard MIMIC hospital-discharge flag without guessing."""

    flag = pd.to_numeric(values, errors='coerce')
    survived = pd.Series(pd.NA, index=values.index, dtype='boolean')
    survived.loc[flag.eq(0)] = True
    survived.loc[flag.eq(1)] = False
    return survived


@dataclass
class FilterCriteria:
    """筛选条件数据类"""
    # 年龄筛选
    age_min: Optional[float] = None
    age_max: Optional[float] = None
    
    # 首次入ICU
    first_icu_stay: Optional[bool] = None
    
    # ICU住院时长（小时）
    los_min: Optional[float] = None
    los_max: Optional[float] = None
    
    # 性别 ('M', 'F', None=不限)
    gender: Optional[str] = None
    
    # 入院类型
    admission_type: Optional[str] = None
    
    # 是否存活至出院（统一为 hospital-discharge endpoint）
    survived: Optional[bool] = None
    
    # Sepsis筛选
    has_sepsis: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() if v is not None}
    
    def __str__(self) -> str:
        conditions = []
        if self.age_min is not None or self.age_max is not None:
            age_str = f"{self.age_min or 0}-{self.age_max or '∞'}岁"
            conditions.append(f"年龄: {age_str}")
        if self.first_icu_stay is not None:
            conditions.append(f"首次入ICU: {'是' if self.first_icu_stay else '否'}")
        if self.los_min is not None or self.los_max is not None:
            los_str = f"{self.los_min or 0}-{self.los_max or '∞'}小时"
            conditions.append(f"住院时长: {los_str}")
        if self.gender is not None:
            conditions.append(f"性别: {self.gender}")
        if self.survived is not None:
            conditions.append(f"存活: {'是' if self.survived else '否'}")
        if self.has_sepsis is not None:
            conditions.append(f"Sepsis: {'是' if self.has_sepsis else '否'}")
        return ", ".join(conditions) if conditions else "无筛选条件"


class PatientFilter:
    """
    患者筛选器 - 支持多种筛选条件
    
    Attributes:
        database: 数据库类型 ('miiv', 'eicu', 'aumc', 'hirid')
        data_path: 数据路径
        demographics: 人口统计学数据（缓存）
    """
    
    def __init__(
        self,
        database: str = 'miiv',
        data_path: Optional[Union[str, Path]] = None,
        verbose: bool = False
    ):
        try:
            self.database = normalize_database_key(database)
        except KeyError as exc:
            raise ValueError(f"不支持的数据库: {database}") from exc
        self.data_path = Path(data_path) if data_path else None
        self.verbose = verbose
        
        # 缓存
        self._demographics: Optional[pd.DataFrame] = None
        self._icustays: Optional[pd.DataFrame] = None
        self._last_criteria: Optional[FilterCriteria] = None
        self._last_result: Optional[pd.DataFrame] = None
        
    def _load_demographics(self) -> pd.DataFrame:
        """加载人口统计学数据"""
        if self._demographics is not None:
            return self._demographics
        
        if self.data_path is None:
            raise ValueError("data_path未设置")
        
        if self.database == 'miiv':
            # MIMIC-IV: 需要合并 patients + icustays + admissions
            self._demographics = self._load_miiv_demographics()
        elif self.database == 'mimic':
            # MIMIC-III: 使用 icustay_id 而非 stay_id
            self._demographics = self._load_mimic3_demographics()
        elif self.database == 'eicu':
            self._demographics = self._load_eicu_demographics()
        elif self.database == 'aumc':
            self._demographics = self._load_aumc_demographics()
        elif self.database == 'hirid':
            self._demographics = self._load_hirid_demographics()
        elif self.database == 'sic':
            self._demographics = self._load_sic_demographics()
        else:
            raise ValueError(f"不支持的数据库: {self.database}")
        
        return self._demographics
    
    def _load_miiv_demographics(self) -> pd.DataFrame:
        """加载MIMIC-IV人口统计学数据"""
        # 加载基础表
        icustays = self._read_table('icustays')
        patients = self._read_table('patients')
        admissions = self._read_table('admissions')
        
        # 合并
        df = icustays.merge(patients, on='subject_id', how='left')
        df = df.merge(admissions, on=['subject_id', 'hadm_id'], how='left')
        
        # 计算年龄（入ICU时的年龄）
        if 'anchor_age' in df.columns:
            # MIMIC-IV 2.0+: 使用anchor_age + (intime.year - anchor_year)
            if 'anchor_year' in df.columns:
                df['intime'] = pd.to_datetime(df['intime'])
                df['age'] = df['anchor_age'] + (df['intime'].dt.year - df['anchor_year'])
            else:
                df['age'] = df['anchor_age']
        elif 'dob' in df.columns:
            # 老版本: 使用dob计算
            df['dob'] = pd.to_datetime(df['dob'])
            df['intime'] = pd.to_datetime(df['intime'])
            df['age'] = (df['intime'] - df['dob']).dt.days / 365.25
        
        # 计算ICU住院时长（小时）
        if 'los' in df.columns:
            df['los_hours'] = df['los'] * 24  # los是天数
        elif 'intime' in df.columns and 'outtime' in df.columns:
            df['intime'] = pd.to_datetime(df['intime'])
            df['outtime'] = pd.to_datetime(df['outtime'])
            df['los_hours'] = (df['outtime'] - df['intime']).dt.total_seconds() / 3600
        
        # Generic first_icu_stay is patient-global. MIMIC exposes no absolute
        # patient ICU ordinal, and this extract has no history-completeness receipt.
        
        # 性别标准化
        if 'gender' in df.columns:
            df['gender'] = df['gender'].str.upper().str[0]  # 'M' or 'F'
        
        # 存活状态
        if 'hospital_expire_flag' in df.columns:
            df['survived'] = _hospital_survival_from_expire_flag(
                df['hospital_expire_flag']
            )
        
        # ID列标准化
        df['patient_id'] = df['stay_id']
        
        return df
    
    def _load_eicu_demographics(self) -> pd.DataFrame:
        """加载eICU人口统计学数据"""
        patient = self._read_table('patient')
        
        # eICU直接有age列，但可能是字符串
        if 'age' in patient.columns:
            # 处理 "> 89" 这样的值
            age_text = patient['age'].astype(str).str.replace(r'\s+', '', regex=True)
            patient['age'] = pd.to_numeric(
                age_text.replace({'>89': '90'}), errors='coerce'
            )
        
        # 住院时长
        if 'unitdischargeoffset' in patient.columns:
            patient['los_hours'] = patient['unitdischargeoffset'] / 60  # 分钟转小时
        
        # unitVisitNumber only orders unit stays within one hospitalization;
        # eICU cannot reliably order separate hospitalizations for one patient.
        if 'unitvisitnumber' in patient.columns:
            visit_number = pd.to_numeric(patient['unitvisitnumber'], errors='coerce')
            first_unit_stay = pd.Series(pd.NA, index=patient.index, dtype='boolean')
            first_unit_stay.loc[visit_number.notna()] = visit_number.eq(1)
            patient['first_unit_stay_within_hospitalization'] = first_unit_stay
        
        # 性别
        if 'gender' in patient.columns:
            patient['gender'] = patient['gender'].str.upper().str[0]
        
        # Generic survived is strictly hospital-discharge survival. Unit-level
        # status remains available under an endpoint-specific name and must not
        # substitute when hospitalDischargeStatus is absent.
        if 'hospitaldischargestatus' in patient.columns:
            status = patient['hospitaldischargestatus'].astype('string').str.strip().str.lower()
            survived = pd.Series(pd.NA, index=patient.index, dtype='boolean')
            survived.loc[status.eq('alive')] = True
            survived.loc[status.eq('expired')] = False
            patient['survived'] = survived
        if 'unitdischargestatus' in patient.columns:
            status = patient['unitdischargestatus'].astype('string').str.strip().str.lower()
            survived = pd.Series(pd.NA, index=patient.index, dtype='boolean')
            survived.loc[status.eq('alive')] = True
            survived.loc[status.eq('expired')] = False
            patient['unit_survived'] = survived
        
        # ID列标准化
        patient['patient_id'] = patient['patientunitstayid']
        
        return patient
    
    def _load_aumc_demographics(self) -> pd.DataFrame:
        """加载AUMC人口统计学数据"""
        admissions = self._read_table('admissions')
        
        # 年龄
        if 'agegroup' in admissions.columns:
            # Preserve the published interval. The midpoint remains available
            # for summaries, but exact inclusion thresholds use the bounds.
            age_bounds = {
                '18-39': (18.0, 39.0),
                '40-49': (40.0, 49.0),
                '50-59': (50.0, 59.0),
                '60-69': (60.0, 69.0),
                '70-79': (70.0, 79.0),
                '80+': (80.0, None),
            }
            bounds = admissions['agegroup'].map(age_bounds)
            admissions['age_lower'] = bounds.map(
                lambda value: value[0] if isinstance(value, tuple) else pd.NA
            )
            admissions['age_upper'] = bounds.map(
                lambda value: value[1] if isinstance(value, tuple) else pd.NA
            )
            admissions['age'] = bounds.map(
                lambda value: (
                    (value[0] + value[1]) / 2
                    if isinstance(value, tuple) and value[1] is not None
                    else (85.0 if isinstance(value, tuple) else pd.NA)
                )
            )
        
        # 住院时长
        if 'admittedat' in admissions.columns and 'dischargedat' in admissions.columns:
            # AUMC时间是毫秒
            admissions['los_hours'] = (
                admissions['dischargedat'] - admissions['admittedat']
            ) / (1000 * 3600)
        
        # admissioncount is the absolute per-patient ICU/MCU admission number;
        # sorting only the current extract would promote a later stay to first.
        if 'admissioncount' in admissions.columns:
            admission_count = pd.to_numeric(
                admissions['admissioncount'], errors='coerce'
            )
            first_stay = pd.Series(pd.NA, index=admissions.index, dtype='boolean')
            first_stay.loc[admission_count.eq(1)] = True
            first_stay.loc[admission_count.gt(1)] = False
            admissions['first_icu_stay'] = first_stay
        
        # 性别
        if 'gender' in admissions.columns:
            gender_map = {'Man': 'M', 'Vrouw': 'F', 'Male': 'M', 'Female': 'F'}
            admissions['gender'] = admissions['gender'].map(gender_map).fillna(
                admissions['gender'].str.upper().str[0]
            )
        
        # destination is survival at ICU/MCU discharge, not hospital discharge.
        # Preserve that endpoint explicitly; generic survived therefore remains
        # unavailable for AmsterdamUMCdb and fails closed when requested.
        if 'destination' in admissions.columns:
            destination = admissions['destination'].astype('string').str.strip().str.lower()
            known = destination.notna() & destination.ne('')
            survived = pd.Series(pd.NA, index=admissions.index, dtype='boolean')
            survived.loc[known] = ~destination.loc[known].str.contains(
                'died|death|deceased|overleden', regex=True
            )
            admissions['icu_survived'] = survived
        
        # ID列标准化
        admissions['patient_id'] = admissions['admissionid']
        
        return admissions
    
    def _load_hirid_demographics(self) -> pd.DataFrame:
        """加载HiRID人口统计学数据"""
        general = self._read_table('general_table')
        
        # HiRID已有age列
        if 'age' not in general.columns and 'admissiontime' in general.columns:
            # 如果没有age，尝试从其他来源获取
            pass
        
        # 住院时长
        if 'admissiontime' in general.columns and 'dischargetime' in general.columns:
            general['admissiontime'] = pd.to_datetime(general['admissiontime'])
            general['dischargetime'] = pd.to_datetime(general['dischargetime'])
            general['los_hours'] = (
                general['dischargetime'] - general['admissiontime']
            ).dt.total_seconds() / 3600
        
        # HiRID assigns a new patient ID to every ICU (re-)admission and does
        # not expose cross-admission patient linkage, so first stay is unknown.
        
        # 性别
        if 'sex' in general.columns:
            general['gender'] = general['sex'].map({'M': 'M', 'F': 'F', 'Male': 'M', 'Female': 'F'})
        
        # ID列标准化
        general['patient_id'] = general['patientid']
        
        return general
    
    def _load_mimic3_demographics(self) -> pd.DataFrame:
        """加载MIMIC-III人口统计学数据"""
        # 加载基础表
        icustays = self._read_table('icustays')
        patients = self._read_table('patients')
        admissions = self._read_table('admissions')
        
        # 合并 - MIMIC-III 使用 icustay_id
        df = icustays.merge(patients, on='subject_id', how='left')
        df = df.merge(admissions, on=['subject_id', 'hadm_id'], how='left')
        
        # 计算年龄（入ICU时的年龄）
        if 'dob' in df.columns and 'intime' in df.columns:
            df['dob'] = pd.to_datetime(df['dob'])
            df['intime'] = pd.to_datetime(df['intime'])
            df['age'] = (df['intime'] - df['dob']).dt.days / 365.25
            # MIMIC-III 对 >89 岁的患者做了脱敏，显示为 ~300岁
            df.loc[df['age'] > 90, 'age'] = 90
        
        # 计算ICU住院时长（小时）
        if 'los' in df.columns:
            df['los_hours'] = df['los'] * 24  # los是天数
        elif 'intime' in df.columns and 'outtime' in df.columns:
            df['intime'] = pd.to_datetime(df['intime'])
            df['outtime'] = pd.to_datetime(df['outtime'])
            df['los_hours'] = (df['outtime'] - df['intime']).dt.total_seconds() / 3600
        
        # Generic first_icu_stay is patient-global. MIMIC exposes no absolute
        # patient ICU ordinal, and this extract has no history-completeness receipt.
        
        # 性别标准化
        if 'gender' in df.columns:
            df['gender'] = df['gender'].str.upper().str[0]  # 'M' or 'F'
        
        # 存活状态
        if 'hospital_expire_flag' in df.columns:
            df['survived'] = _hospital_survival_from_expire_flag(
                df['hospital_expire_flag']
            )
        
        # ID列标准化 - MIMIC-III 使用 icustay_id
        df['patient_id'] = df['icustay_id']
        
        return df
    
    def _load_sic_demographics(self) -> pd.DataFrame:
        """加载SICdb人口统计学数据"""
        # SICdb 使用 cases 表
        cases = self._read_table('cases')
        
        columns = {str(column).lower(): column for column in cases.columns}

        # SICdb publishes rounded age in AgeOnAdmission.
        age_column = columns.get('ageonadmission') or columns.get('age')
        if age_column is not None:
            cases['age'] = pd.to_numeric(cases[age_column], errors='coerce')

        # Both offsets are seconds from primary MetaVision admission. ICU LOS
        # starts at ICUOffset, so preceding surgery must be removed.
        time_of_stay = columns.get('timeofstay')
        icu_offset = columns.get('icuoffset')
        if time_of_stay is not None and icu_offset is not None:
            los_seconds = (
                pd.to_numeric(cases[time_of_stay], errors='coerce')
                - pd.to_numeric(cases[icu_offset], errors='coerce')
            )
            cases['los_hours'] = los_seconds.where(los_seconds >= 0) / 3600.0
        
        # OffsetAfterFirstAdmission has absolute semantics: zero identifies the
        # first admission, positive values identify readmissions even when the
        # first case is absent from the current extract. Negative/missing values
        # remain unknown. Duplicate zeroes for one patient are ambiguous.
        patient_id = columns.get('patientid')
        first_admission_offset = columns.get('offsetafterfirstadmission')
        if patient_id is not None and first_admission_offset is not None:
            patient = cases[patient_id]
            offset = pd.to_numeric(cases[first_admission_offset], errors='coerce')
            first_stay = pd.Series(pd.NA, index=cases.index, dtype='boolean')
            first_stay.loc[offset.gt(0)] = False
            zero_with_patient = offset.eq(0) & patient.notna()
            if zero_with_patient.any():
                zero_count = zero_with_patient.groupby(patient).transform('sum')
                first_stay.loc[zero_with_patient & zero_count.eq(1)] = True
            cases['first_icu_stay'] = first_stay
        
        # 性别
        if 'Sex' in cases.columns:
            cases['gender'] = cases['Sex'].map({0: 'F', 1: 'M', 'Female': 'F', 'Male': 'M', 'F': 'F', 'M': 'M'})
        elif 'sex' in cases.columns:
            cases['gender'] = cases['sex'].map({0: 'F', 1: 'M', 'Female': 'F', 'Male': 'M', 'F': 'F', 'M': 'M'})
        
        # HospitalDischargeType is the hospital-discharge endpoint. OffsetOfDeath
        # is one-year follow-up and must not be substituted for discharge survival.
        discharge_column = columns.get('hospitaldischargetype')
        if discharge_column is not None:
            discharge = cases[discharge_column]
            numeric = pd.to_numeric(discharge, errors='coerce')
            survived = pd.Series(pd.NA, index=cases.index, dtype='boolean')
            survived.loc[numeric.eq(2026)] = True
            survived.loc[numeric.eq(2028)] = False
            text = discharge.astype(str).str.strip().str.lower()
            survived.loc[text.eq('survived')] = True
            survived.loc[text.eq('deceased')] = False
            cases['survived'] = survived
        
        # ID列标准化 - SICdb 使用 CaseID
        case_id = columns.get('caseid')
        if case_id is not None:
            cases['patient_id'] = cases[case_id]
        
        return cases
    
    def _read_table(self, table_name: str) -> pd.DataFrame:
        """读取数据表，支持多种目录结构"""
        if self.data_path is None:
            raise ValueError("data_path未设置")
        
        # 搜索路径优先级: 根目录 → 已知子目录(icu/, hosp/) → 任意子目录
        search_dirs = [self.data_path]
        for sub in ['icu', 'hosp']:
            sub_path = self.data_path / sub
            if sub_path.is_dir():
                search_dirs.append(sub_path)
        
        for search_dir in search_dirs:
            # 尝试多种格式
            for ext in ['.parquet', '.csv.gz', '.csv']:
                path = search_dir / f"{table_name}{ext}"
                if path.exists():
                    if ext == '.parquet':
                        return pd.read_parquet(path)
                    else:
                        return pd.read_csv(path)
            
            # 尝试目录格式（分片parquet）
            dir_path = search_dir / table_name
            if dir_path.is_dir():
                parquet_files = list(dir_path.glob('*.parquet'))
                if parquet_files:
                    dfs = [pd.read_parquet(f) for f in parquet_files]
                    return pd.concat(dfs, ignore_index=True)
            
            # 尝试分桶目录
            bucket_dir = search_dir / f"{table_name}_bucket"
            if bucket_dir.is_dir():
                parquet_files = list(bucket_dir.rglob('*.parquet'))
                if parquet_files:
                    dfs = [pd.read_parquet(f) for f in parquet_files[:50]]  # 限制读取量
                    return pd.concat(dfs, ignore_index=True)
        
        # 最后尝试: rglob 搜索任意深度
        for ext in ['.parquet', '.csv.gz', '.csv']:
            found = list(self.data_path.rglob(f"{table_name}{ext}"))
            if found:
                path = found[0]
                if ext == '.parquet':
                    return pd.read_parquet(path)
                else:
                    return pd.read_csv(path)
        
        raise FileNotFoundError(f"找不到表 {table_name} (路径: {self.data_path})")
    
    def filter(
        self,
        age_min: Optional[float] = None,
        age_max: Optional[float] = None,
        first_icu_stay: Optional[bool] = None,
        los_min: Optional[float] = None,
        los_max: Optional[float] = None,
        gender: Optional[str] = None,
        survived: Optional[bool] = None,
        has_sepsis: Optional[bool] = None,
        return_dataframe: bool = False,
    ) -> Union[List[int], pd.DataFrame]:
        """
        根据条件筛选患者
        
        Args:
            age_min: 最小年龄
            age_max: 最大年龄
            first_icu_stay: 是否首次入ICU
            los_min: 最短住院时长（小时）
            los_max: 最长住院时长（小时）
            gender: 性别 ('M' 或 'F')
            survived: 是否存活至医院出院（hospital-discharge endpoint）
            has_sepsis: 是否有Sepsis（需要额外加载诊断数据）
            return_dataframe: 是否返回完整DataFrame而非仅ID列表
        
        Returns:
            患者ID列表或DataFrame
        """
        criteria = FilterCriteria(
            age_min=age_min, age_max=age_max,
            first_icu_stay=first_icu_stay,
            los_min=los_min, los_max=los_max,
            gender=gender, survived=survived,
            has_sepsis=has_sepsis
        )
        
        # 加载数据
        df = self._load_demographics()
        original_count = len(df)
        self._last_original_count = original_count
        
        if self.verbose:
            logger.info(f"开始筛选: 原始患者数 {original_count}")
        
        # 应用筛选条件
        mask = pd.Series([True] * len(df), index=df.index)

        requested_columns = {
            'age': ('age', age_min is not None or age_max is not None),
            'first_icu_stay': ('first_icu_stay', first_icu_stay is not None),
            'los': ('los_hours', los_min is not None or los_max is not None),
            'gender': ('gender', gender is not None),
            'survived': ('survived', survived is not None),
        }
        for criterion, (column, requested) in requested_columns.items():
            if requested and (column not in df.columns or df[column].notna().sum() == 0):
                raise PatientFilterCriterionError(
                    'patient_filter_criterion_unavailable',
                    criterion,
                    f"Requested criterion {criterion!r} is unavailable for {self.database!r}.",
                )
        
        # 年龄筛选
        if (
            self.database == 'aumc'
            and (age_min is not None or age_max is not None)
            and {'age_lower', 'age_upper'}.issubset(df.columns)
        ):
            lower = pd.to_numeric(df['age_lower'], errors='coerce')
            upper = pd.to_numeric(df['age_upper'], errors='coerce')
            if age_min is not None:
                split = lower.lt(age_min) & (upper.isna() | upper.ge(age_min))
                if split.any():
                    raise PatientFilterCriterionError(
                        'patient_filter_grouped_age_indeterminate',
                        'age',
                        f"age_min={age_min} splits an AmsterdamUMCdb age band.",
                    )
                mask &= lower.ge(age_min).fillna(False)
            if age_max is not None:
                split = lower.le(age_max) & (upper.isna() | upper.gt(age_max))
                if split.any():
                    raise PatientFilterCriterionError(
                        'patient_filter_grouped_age_indeterminate',
                        'age',
                        f"age_max={age_max} splits an AmsterdamUMCdb age band.",
                    )
                mask &= upper.le(age_max).fillna(False)
        else:
            if age_min is not None:
                mask &= df['age'] >= age_min
            if age_max is not None:
                mask &= df['age'] <= age_max
        
        # 首次入ICU
        if first_icu_stay is not None:
            mask &= df['first_icu_stay'] == first_icu_stay
        
        # 住院时长
        if los_min is not None:
            mask &= df['los_hours'] >= los_min
        if los_max is not None:
            mask &= df['los_hours'] <= los_max
        
        # 性别
        if gender is not None:
            mask &= df['gender'].str.upper() == gender.upper()
        
        # 存活状态
        if survived is not None:
            mask &= df['survived'] == survived
        
        # Sepsis筛选（需要额外处理）
        if has_sepsis is not None:
            sepsis_ids = self._get_sepsis_patients()
            if has_sepsis:
                mask &= df['patient_id'].isin(sepsis_ids)
            else:
                mask &= ~df['patient_id'].isin(sepsis_ids)
        
        # 应用筛选
        result = df[mask].copy()
        
        # 保存结果
        self._last_criteria = criteria
        self._last_result = result
        
        if self.verbose:
            logger.info(f"筛选完成: {len(result)}/{original_count} ({len(result)/original_count*100:.1f}%)")
        
        if return_dataframe:
            return result
        else:
            return result['patient_id'].tolist()
    
    def _get_sepsis_patients(self) -> set:
        """获取 Sepsis 患者 ID 集合。

        使用 EasyICU 的 Sepsis-3 定义（疑似感染 susp_inf + SOFA），而非 ICD
        诊断编码。Sepsis-3 是推导出来的概念，没有便宜的查表捷径：本函数会触发
        一次完整的 load_sepsis3 推导，这是保证队列定义正确的必要代价。ICD 编码
        的脓毒症（A40/A41 等）只覆盖真实 Sepsis-3 队列的一个子集，不能作为代理。
        """
        try:
            from easyicu.api import load_sepsis3
            sep3 = load_sepsis3(database=self.database, data_path=self.data_path)
        except Exception as e:
            raise PatientFilterCriterionError(
                'patient_filter_sepsis_unascertainable',
                'has_sepsis',
                'Sepsis-3 derivation failed; the cohort cannot encode unknown as negative.',
            ) from e

        if not isinstance(sep3, pd.DataFrame) or sep3.empty:
            return set()

        id_candidates = ['stay_id', 'icustay_id', 'patientunitstayid',
                         'admissionid', 'patientid', 'CaseID']
        id_col = next((c for c in id_candidates if c in sep3.columns), None)
        if id_col is None:
            raise PatientFilterCriterionError(
                'patient_filter_sepsis_identifier_unavailable',
                'has_sepsis',
                'The Sepsis-3 result has no supported ICU-stay identifier.',
            )

        value_candidates = ['sep3', 'sep3_sofa2', 'sep3_sofa1']
        value_col = next((c for c in value_candidates if c in sep3.columns), None)
        if value_col is None:
            # 概念结果无显式标签列：出现在结果中即视为 Sepsis-3 阳性
            return set(sep3[id_col].dropna().unique())

        vals = pd.to_numeric(sep3[value_col], errors='coerce')
        if vals.notna().any():
            mask = vals > 0
        else:
            mask = sep3[value_col].astype(str).str.strip().str.lower().isin(
                {'1', 'true', 't', 'yes', 'y'})
        return set(sep3.loc[mask.fillna(False), id_col].dropna().unique())
    
    def get_filter_summary(self) -> Dict[str, Any]:
        """获取筛选结果摘要"""
        if self._last_result is None:
            return {"error": "未执行筛选"}
        
        df = self._last_result
        summary = {
            "筛选条件": str(self._last_criteria),
            "筛选后患者数": len(df),
        }
        
        # 统计信息
        if 'age' in df.columns:
            summary["年龄分布"] = {
                "均值": round(df['age'].mean(), 1),
                "中位数": round(df['age'].median(), 1),
                "范围": f"{df['age'].min():.0f}-{df['age'].max():.0f}"
            }
        
        if 'los_hours' in df.columns:
            summary["住院时长(小时)"] = {
                "均值": round(df['los_hours'].mean(), 1),
                "中位数": round(df['los_hours'].median(), 1),
            }
        
        if 'gender' in df.columns:
            gender_dist = df['gender'].value_counts(normalize=True)
            summary["性别分布"] = {
                k: f"{v*100:.1f}%" for k, v in gender_dist.items()
            }
        
        if 'survived' in df.columns:
            survival_rate = df['survived'].mean() * 100
            summary["存活率"] = f"{survival_rate:.1f}%"
        
        return summary
    
    def get_cohort_comparison(
        self,
        group_by: str = 'survived',
        custom_groups: Optional[Dict[str, List]] = None
    ) -> pd.DataFrame:
        """
        获取队列对比统计
        
        Args:
            group_by: 分组依据列名 ('survived', 'gender', 'first_icu_stay' 等)
            custom_groups: 自定义分组 {组名: [患者ID列表]}
        
        Returns:
            对比统计DataFrame
        """
        if self._last_result is None:
            raise ValueError("请先执行filter()筛选")
        
        df = self._last_result
        
        if custom_groups is not None:
            # 使用自定义分组
            df = df.copy()
            df['group'] = 'Other'
            for group_name, patient_ids in custom_groups.items():
                df.loc[df['patient_id'].isin(patient_ids), 'group'] = group_name
            group_col = 'group'
        else:
            group_col = group_by
        
        if group_col not in df.columns:
            raise ValueError(f"列 {group_col} 不存在")
        
        # 统计各组
        stats = []
        for group_name, group_df in df.groupby(group_col):
            stat = {
                '组别': group_name,
                '患者数': len(group_df),
                '占比': f"{len(group_df)/len(df)*100:.1f}%",
            }
            
            if 'age' in group_df.columns:
                stat['年龄(均值±SD)'] = f"{group_df['age'].mean():.1f}±{group_df['age'].std():.1f}"
            
            if 'los_hours' in group_df.columns:
                stat['住院时长(中位数)'] = f"{group_df['los_hours'].median():.1f}h"
            
            if 'gender' in group_df.columns:
                male_pct = (group_df['gender'] == 'M').mean() * 100
                stat['男性占比'] = f"{male_pct:.1f}%"
            
            if 'survived' in group_df.columns:
                stat['存活率'] = f"{group_df['survived'].mean()*100:.1f}%"
            
            stats.append(stat)
        
        return pd.DataFrame(stats)


def filter_patients(
    database: str = 'miiv',
    data_path: Optional[Union[str, Path]] = None,
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
    first_icu_stay: Optional[bool] = None,
    los_min: Optional[float] = None,
    los_max: Optional[float] = None,
    gender: Optional[str] = None,
    survived: Optional[bool] = None,
    has_sepsis: Optional[bool] = None,
    verbose: bool = False,
    return_dataframe: bool = False,
) -> Union[List[int], pd.DataFrame]:
    """
    便捷函数：根据条件筛选患者
    
    Args:
        database: 数据库类型
        data_path: 数据路径
        age_min: 最小年龄
        age_max: 最大年龄
        first_icu_stay: 是否首次入ICU
        los_min: 最短住院时长（小时）
        los_max: 最长住院时长（小时）
        gender: 性别 ('M' 或 'F')
        survived: 是否存活出院
        has_sepsis: 是否有Sepsis
        verbose: 显示详细信息
        return_dataframe: 返回DataFrame而非ID列表
    
    Returns:
        患者ID列表或DataFrame
    
    Examples:
        >>> # 筛选18-80岁首次入ICU的患者
        >>> ids = filter_patients(
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     age_min=18, age_max=80,
        ...     first_icu_stay=True
        ... )
        >>> print(f"筛选到 {len(ids)} 名患者")
        >>>
        >>> # 筛选Sepsis患者
        >>> sepsis_ids = filter_patients(
        ...     database='miiv',
        ...     data_path='/path/to/data',
        ...     has_sepsis=True
        ... )
    """
    pf = PatientFilter(database=database, data_path=data_path, verbose=verbose)
    return pf.filter(
        age_min=age_min, age_max=age_max,
        first_icu_stay=first_icu_stay,
        los_min=los_min, los_max=los_max,
        gender=gender, survived=survived,
        has_sepsis=has_sepsis,
        return_dataframe=return_dataframe
    )


def get_cohort_stats(
    patient_ids: List[int],
    database: str = 'miiv',
    data_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    获取患者队列统计信息
    
    Args:
        patient_ids: 患者ID列表
        database: 数据库类型
        data_path: 数据路径
    
    Returns:
        统计信息字典
    """
    pf = PatientFilter(database=database, data_path=data_path)
    df = pf._load_demographics()
    df = df[df['patient_id'].isin(patient_ids)]
    
    stats = {
        "患者数": len(df),
    }
    
    if 'age' in df.columns:
        stats["年龄"] = {
            "均值": round(df['age'].mean(), 1),
            "标准差": round(df['age'].std(), 1),
            "中位数": round(df['age'].median(), 1),
            "Q1": round(df['age'].quantile(0.25), 1),
            "Q3": round(df['age'].quantile(0.75), 1),
        }
    
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        stats["性别"] = gender_counts.to_dict()
    
    if 'los_hours' in df.columns:
        stats["住院时长(小时)"] = {
            "均值": round(df['los_hours'].mean(), 1),
            "中位数": round(df['los_hours'].median(), 1),
        }
    
    if 'survived' in df.columns:
        stats["存活率"] = f"{df['survived'].mean()*100:.1f}%"
    
    if 'first_icu_stay' in df.columns:
        stats["首次入ICU占比"] = f"{df['first_icu_stay'].mean()*100:.1f}%"
    
    return stats
