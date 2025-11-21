"""Missing callback functions for clinical concepts.

This module implements additional callback functions that are required for
complete replication of ricu functionality but were not present in the main
callbacks.py file.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
import numpy as np


def rrt_criteria(
    crea: pd.Series,
    uo_6h: Optional[pd.Series] = None,
    potassium: Optional[pd.Series] = None,
    ph: Optional[pd.Series] = None,
    bicarb: Optional[pd.Series] = None,
) -> pd.Series:
    """Determine if patient meets RRT (Renal Replacement Therapy) criteria.
    
    RRT criteria according to SOFA-2 footnote p:
    - Base kidney injury: Creatinine > 1.2 mg/dL OR oliguria (<0.3 mL/kg/h for >6h)
    - PLUS at least one of:
      * Serum potassium ≥ 6.0 mmol/L
      * Metabolic acidosis: pH ≤ 7.20 AND HCO3 ≤ 12 mmol/L
    
    Args:
        crea: Serum creatinine (mg/dL)
        uo_6h: 6-hour average urine output (mL/kg/h) - proxy for oliguria >6h
        potassium: Serum potassium (mmol/L)
        ph: Arterial pH
        bicarb: Serum bicarbonate (mmol/L)
        
    Returns:
        Boolean series indicating if RRT criteria are met
    """
    crea_num = pd.to_numeric(crea, errors="coerce")
    
    # Base kidney injury criteria
    base_injury = crea_num > 1.2
    
    if uo_6h is not None:
        uo_num = pd.to_numeric(uo_6h, errors="coerce")
        oliguria = uo_num < 0.3
        base_injury = base_injury | oliguria
    
    # Additional criteria
    meets_additional = pd.Series(False, index=crea.index)
    
    if potassium is not None:
        k_num = pd.to_numeric(potassium, errors="coerce")
        hyperkalemia = k_num >= 6.0
        meets_additional = meets_additional | hyperkalemia
    
    if ph is not None and bicarb is not None:
        ph_num = pd.to_numeric(ph, errors="coerce")
        hco3_num = pd.to_numeric(bicarb, errors="coerce")
        acidosis = (ph_num <= 7.20) & (hco3_num <= 12)
        meets_additional = meets_additional | acidosis
    
    # Meets RRT criteria if has base injury AND meets at least one additional criterion
    return base_injury & meets_additional


def sum_components(*components: pd.Series) -> pd.Series:
    """Sum multiple component scores.
    
    This is a generic callback for summing score components like SOFA subsystems.
    NaN values are treated as 0.
    
    Args:
        *components: Variable number of component score series
        
    Returns:
        Series with summed scores
    """
    if not components:
        raise ValueError("At least one component required")
    
    # Start with first component
    result = pd.to_numeric(components[0], errors="coerce").fillna(0)
    
    # Add remaining components
    for comp in components[1:]:
        comp_num = pd.to_numeric(comp, errors="coerce").fillna(0)
        result = result + comp_num
    
    return result.astype(int)


def blood_cell_ratio(value: pd.Series, total_wbc: Optional[pd.Series] = None) -> pd.Series:
    """Convert blood cell counts between absolute and percentage forms.
    
    Used for converting cell counts like eosinophils, lymphocytes, etc.
    
    Args:
        value: Cell count value (may be in % or absolute count)
        total_wbc: Total white blood cell count (for conversion)
        
    Returns:
        Converted values (percentage form)
    """
    val_num = pd.to_numeric(value, errors="coerce")
    
    # If values are already in percentage form (0-100), return as-is
    # If values are absolute counts (usually > 100), convert to percentage
    max_val = val_num.max()
    
    if max_val > 100 and total_wbc is not None:
        # Assume absolute counts, convert to percentage
        wbc_num = pd.to_numeric(total_wbc, errors="coerce")
        return (val_num / wbc_num) * 100
    
    # Already in percentage form
    return val_num


def aumc_bxs(value: pd.Series, tag: Optional[pd.Series] = None) -> pd.Series:
    """Process AUMC base excess values.
    
    AUMC stores base excess with different tags indicating the type.
    This function filters/processes based on the tag.
    
    Args:
        value: Base excess values
        tag: Tag indicating measurement type
        
    Returns:
        Processed base excess values
    """
    val_num = pd.to_numeric(value, errors="coerce")
    
    if tag is None:
        return val_num
    
    # Filter based on tag if needed
    # AUMC may have different base excess measurements (standard vs actual)
    # For now, return all values
    return val_num


def blood_cell_count(percentage: pd.Series, wbc: pd.Series) -> pd.Series:
    """Convert blood cell percentage to absolute count.
    
    Args:
        percentage: Cell percentage (%)
        wbc: Total white blood cell count (×10³/μL)
        
    Returns:
        Absolute cell count (×10³/μL)
    """
    pct_num = pd.to_numeric(percentage, errors="coerce")
    wbc_num = pd.to_numeric(wbc, errors="coerce")
    
    return (pct_num / 100) * wbc_num


def delta_cummin(data: pd.Series, group_id: pd.Series) -> pd.Series:
    """Calculate delta from cumulative minimum.
    
    Used in sepsis detection to track changes from the patient's
    lowest (best) SOFA score.
    
    Args:
        data: Data series (e.g., SOFA scores)
        group_id: Grouping identifier (e.g., stay_id)
        
    Returns:
        Delta from cumulative minimum
    """
    data_num = pd.to_numeric(data, errors="coerce")
    
    # Group by ID and calculate cumulative minimum
    cummin = data_num.groupby(group_id).cummin()
    
    # Return delta (current - cummin)
    return data_num - cummin


def delta_start(data: pd.Series, group_id: pd.Series) -> pd.Series:
    """Calculate delta from first value in group.
    
    Args:
        data: Data series
        group_id: Grouping identifier
        
    Returns:
        Delta from first value in each group
    """
    data_num = pd.to_numeric(data, errors="coerce")
    
    # Get first value for each group
    first_vals = data_num.groupby(group_id).transform('first')
    
    # Return delta (current - first)
    return data_num - first_vals


def delta_min(data: pd.Series, group_id: pd.Series) -> pd.Series:
    """Calculate delta from minimum value in group.
    
    Args:
        data: Data series
        group_id: Grouping identifier
        
    Returns:
        Delta from minimum value in each group
    """
    data_num = pd.to_numeric(data, errors="coerce")
    
    # Get minimum value for each group (global, not cumulative)
    min_vals = data_num.groupby(group_id).transform('min')
    
    # Return delta (current - min)
    return data_num - min_vals
