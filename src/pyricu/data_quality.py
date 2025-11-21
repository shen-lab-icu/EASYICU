"""Data quality validation and reporting utilities.

This module provides comprehensive data quality checks for ICU data,
including missing value analysis, range validation, and consistency checks.
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import pandas as pd
import numpy as np

class DataQualityValidator:
    """Validate data quality for ICU datasets."""
    
    # Expected ranges for common clinical variables
    CLINICAL_RANGES = {
        # Vital signs
        'hr': (0, 300),           # Heart rate (bpm)
        'sbp': (0, 300),          # Systolic BP (mmHg)
        'dbp': (0, 200),          # Diastolic BP (mmHg)
        'mbp': (0, 250),          # Mean BP (mmHg)
        'resp': (0, 100),         # Respiratory rate (/min)
        'temp': (25, 45),         # Temperature (°C)
        'spo2': (0, 100),         # SpO2 (%)
        'o2sat': (0, 100),        # O2 saturation (%)
        
        # Laboratory values
        'glucose': (0, 1000),     # Glucose (mg/dL)
        'lactate': (0, 50),       # Lactate (mmol/L)
        'creatinine': (0, 30),    # Creatinine (mg/dL)
        'bilirubin': (0, 100),    # Bilirubin (mg/dL)
        'wbc': (0, 500),          # WBC count (x10^9/L)
        'platelets': (0, 2000),   # Platelet count (x10^9/L)
        'hemoglobin': (0, 25),    # Hemoglobin (g/dL)
        'hematocrit': (0, 100),   # Hematocrit (%)
        'sodium': (100, 200),     # Sodium (mEq/L)
        'potassium': (0, 15),     # Potassium (mEq/L)
        'chloride': (50, 150),    # Chloride (mEq/L)
        'bicarbonate': (0, 100),  # Bicarbonate (mEq/L)
        'bun': (0, 300),          # BUN (mg/dL)
        'calcium': (0, 20),       # Calcium (mg/dL)
        'magnesium': (0, 10),     # Magnesium (mg/dL)
        'phosphate': (0, 20),     # Phosphate (mg/dL)
        'albumin': (0, 10),       # Albumin (g/dL)
        
        # Blood gas
        'ph': (6.5, 8.0),         # pH
        'pco2': (10, 150),        # PaCO2 (mmHg)
        'po2': (10, 700),         # PaO2 (mmHg)
        'fio2': (0, 1),           # FiO2 (fraction)
        
        # Scores
        'gcs': (3, 15),           # Glasgow Coma Scale
        'sofa': (0, 24),          # SOFA score
        'sirs': (0, 4),           # SIRS score
        'qsofa': (0, 3),          # qSOFA score
        'apache': (0, 299),       # APACHE score
        'saps': (0, 163),         # SAPS score
        
        # Other
        'weight': (0, 500),       # Weight (kg)
        'height': (0, 300),       # Height (cm)
        'bmi': (5, 100),          # BMI
        'age': (0, 120),          # Age (years)
    }
    
    @classmethod
    def validate_ranges(
        cls,
        data: pd.DataFrame,
        variable: str,
        custom_range: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        """Validate if values are within expected ranges.
        
        Args:
            data: DataFrame with variable data
            variable: Variable name to check
            custom_range: Custom (min, max) range, overrides defaults
            
        Returns:
            Dict with validation results
        """
        if variable not in data.columns:
            return {'error': f'Variable {variable} not found'}
        
        values = data[variable].dropna()
        
        if len(values) == 0:
            return {'error': 'No non-NA values'}
        
        # Get expected range
        if custom_range:
            min_val, max_val = custom_range
        elif variable.lower() in cls.CLINICAL_RANGES:
            min_val, max_val = cls.CLINICAL_RANGES[variable.lower()]
        else:
            # No range defined
            return {
                'variable': variable,
                'n_values': len(values),
                'min': values.min(),
                'max': values.max(),
                'range_defined': False,
            }
        
        # Check for out-of-range values
        out_of_range = (values < min_val) | (values > max_val)
        n_out_of_range = out_of_range.sum()
        
        return {
            'variable': variable,
            'n_values': len(values),
            'expected_min': min_val,
            'expected_max': max_val,
            'actual_min': values.min(),
            'actual_max': values.max(),
            'n_out_of_range': n_out_of_range,
            'pct_out_of_range': (n_out_of_range / len(values)) * 100,
            'out_of_range_values': values[out_of_range].tolist() if n_out_of_range > 0 and n_out_of_range < 20 else [],
        }
    
    @classmethod
    def check_missing_data(
        cls,
        data: pd.DataFrame,
        id_cols: Optional[List[str]] = None,
        index_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Analyze missing data patterns.
        
        Args:
            data: DataFrame to analyze
            id_cols: ID columns (excluded from analysis)
            index_col: Time index column (excluded from analysis)
            
        Returns:
            DataFrame with missing data statistics per variable
        """
        id_cols = id_cols or []
        meta_cols = id_cols + ([index_col] if index_col else [])
        data_cols = [c for c in data.columns if c not in meta_cols]
        
        missing_stats = []
        
        for col in data_cols:
            n_missing = data[col].isna().sum()
            n_total = len(data)
            pct_missing = (n_missing / n_total) * 100
            
            missing_stats.append({
                'variable': col,
                'n_total': n_total,
                'n_missing': n_missing,
                'n_present': n_total - n_missing,
                'pct_missing': pct_missing,
                'pct_present': 100 - pct_missing,
            })
        
        return pd.DataFrame(missing_stats).sort_values('pct_missing', ascending=False)
    
    @classmethod
    def check_duplicates(
        cls,
        data: pd.DataFrame,
        subset: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Check for duplicate rows.
        
        Args:
            data: DataFrame to check
            subset: Columns to check for duplicates (None = all columns)
            
        Returns:
            Dict with duplicate statistics
        """
        if subset:
            duplicates = data.duplicated(subset=subset, keep=False)
        else:
            duplicates = data.duplicated(keep=False)
        
        n_duplicates = duplicates.sum()
        
        return {
            'n_total_rows': len(data),
            'n_duplicate_rows': n_duplicates,
            'n_unique_rows': len(data) - n_duplicates,
            'pct_duplicates': (n_duplicates / len(data)) * 100,
            'duplicate_rows': data[duplicates].head(10) if n_duplicates > 0 else pd.DataFrame(),
        }
    
    @classmethod
    def check_time_consistency(
        cls,
        data: pd.DataFrame,
        id_cols: List[str],
        index_col: str,
    ) -> Dict[str, Any]:
        """Check time consistency within patient stays.
        
        Args:
            data: DataFrame with time-indexed data
            id_cols: ID columns
            index_col: Time index column
            
        Returns:
            Dict with time consistency results
        """
        if not pd.api.types.is_datetime64_any_dtype(data[index_col]):
            return {'error': f'{index_col} is not datetime type'}
        
        issues = []
        
        for id_vals, group in data.groupby(id_cols):
            times = group[index_col].sort_values()
            
            # Check for negative time differences
            diffs = times.diff()
            negative_diffs = diffs[diffs < pd.Timedelta(0)]
            
            if len(negative_diffs) > 0:
                issues.append({
                    'id': id_vals if isinstance(id_vals, tuple) else (id_vals,),
                    'issue': 'negative_time_diff',
                    'n_occurrences': len(negative_diffs),
                })
            
            # Check for very large gaps (>30 days)
            large_gaps = diffs[diffs > pd.Timedelta(days=30)]
            
            if len(large_gaps) > 0:
                issues.append({
                    'id': id_vals if isinstance(id_vals, tuple) else (id_vals,),
                    'issue': 'large_time_gap',
                    'n_occurrences': len(large_gaps),
                    'max_gap': large_gaps.max(),
                })
        
        return {
            'n_patients': data[id_cols[0]].nunique(),
            'n_time_issues': len(issues),
            'issues': issues[:10],  # Return first 10
        }
    
    @classmethod
    def comprehensive_report(
        cls,
        data: pd.DataFrame,
        id_cols: Optional[List[str]] = None,
        index_col: Optional[str] = None,
        output_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive data quality report.
        
        Args:
            data: DataFrame to analyze
            id_cols: ID columns
            index_col: Time index column
            output_file: Optional file path to save report
            
        Returns:
            Dict with complete quality report
        """
        id_cols = id_cols or []
        
        report = {
            'basic_info': {
                'n_rows': len(data),
                'n_columns': len(data.columns),
                'n_patients': data[id_cols[0]].nunique() if id_cols else None,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
            },
            'missing_data': cls.check_missing_data(data, id_cols, index_col).to_dict('records'),
            'duplicates': cls.check_duplicates(data, subset=id_cols + [index_col] if id_cols and index_col else None),
        }
        
        # Range validation for known variables
        meta_cols = id_cols + ([index_col] if index_col else [])
        data_cols = [c for c in data.columns if c not in meta_cols]
        
        range_validations = []
        for col in data_cols:
            if col.lower() in cls.CLINICAL_RANGES:
                validation = cls.validate_ranges(data, col)
                range_validations.append(validation)
        
        report['range_validation'] = range_validations
        
        # Time consistency
        if id_cols and index_col and index_col in data.columns:
            report['time_consistency'] = cls.check_time_consistency(data, id_cols, index_col)
        
        # Variable types
        report['variable_types'] = {
            'numeric': data.select_dtypes(include=[np.number]).columns.tolist(),
            'datetime': data.select_dtypes(include=['datetime64']).columns.tolist(),
            'categorical': data.select_dtypes(include=['object', 'category']).columns.tolist(),
        }
        
        # Save report if requested
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            import json
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"Quality report saved to {output_file}")
        
        return report

def validate_data_quality(
    data: pd.DataFrame,
    concept_name: str,
    *,
    check_missing: bool = True,
    check_ranges: bool = True,
    check_duplicates: bool = True,
    id_cols: Optional[List[str]] = None,
    index_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate data quality report for a specific concept.
    
    Args:
        data: DataFrame to validate
        concept_name: Name of the clinical concept
        check_missing: Whether to check missing values
        check_ranges: Whether to check value ranges
        check_duplicates: Whether to check duplicates
        id_cols: ID columns
        index_col: Time index column
        
    Returns:
        Dict with quality metrics
    """
    report = {
        'concept': concept_name,
        'total_rows': len(data),
    }
    
    if check_missing:
        missing_df = DataQualityValidator.check_missing_data(data, id_cols, index_col)
        report['missing_values'] = missing_df.to_dict('records')
    
    if check_ranges:
        # Check ranges for value column
        value_col = [c for c in data.columns if 'value' in c.lower() or c == concept_name]
        if value_col:
            validation = DataQualityValidator.validate_ranges(data, value_col[0])
            report['range_validation'] = validation
    
    if check_duplicates:
        if id_cols and index_col:
            dup_check = DataQualityValidator.check_duplicates(
                data, subset=id_cols + [index_col]
            )
            report['duplicates'] = dup_check
    
    return report

def print_quality_summary(report: Dict[str, Any]) -> None:
    """Print a human-readable summary of quality report.
    
    Args:
        report: Quality report from comprehensive_report()
    """
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    
    # Basic info
    if 'basic_info' in report:
        info = report['basic_info']
        print(f"\nBasic Information:")
        print(f"  Rows: {info['n_rows']:,}")
        print(f"  Columns: {info['n_columns']}")
        if info['n_patients']:
            print(f"  Patients: {info['n_patients']:,}")
        print(f"  Memory: {info['memory_usage_mb']:.2f} MB")
    
    # Missing data
    if 'missing_data' in report and report['missing_data']:
        print(f"\nMissing Data (Top 5):")
        for item in report['missing_data'][:5]:
            print(f"  {item['variable']}: {item['pct_missing']:.1f}% missing")
    
    # Duplicates
    if 'duplicates' in report:
        dup = report['duplicates']
        print(f"\nDuplicates:")
        print(f"  {dup['n_duplicate_rows']} duplicate rows ({dup['pct_duplicates']:.1f}%)")
    
    # Range validation
    if 'range_validation' in report and report['range_validation']:
        out_of_range = [v for v in report['range_validation'] 
                       if v.get('n_out_of_range', 0) > 0]
        if out_of_range:
            print(f"\nOut-of-Range Values:")
            for item in out_of_range[:5]:
                print(f"  {item['variable']}: {item['n_out_of_range']} values "
                     f"({item['pct_out_of_range']:.1f}%)")
    
    # Time consistency
    if 'time_consistency' in report:
        tc = report['time_consistency']
        if tc.get('n_time_issues', 0) > 0:
            print(f"\nTime Consistency Issues:")
            print(f"  {tc['n_time_issues']} patients with time issues")
    
    print("=" * 60)
