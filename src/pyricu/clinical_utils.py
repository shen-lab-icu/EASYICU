"""Clinical utility functions (R ricu callback-itm.R).

Provides clinical assessment and calculation functions.
"""

from __future__ import annotations

from typing import Union, Optional
import pandas as pd
import numpy as np


def avpu(x: Union[pd.Series, str, int]) -> Union[pd.Series, str]:
    """AVPU consciousness level assessment (R ricu avpu).
    
    AVPU is a simplified consciousness assessment:
    - A: Alert
    - V: Voice
    - P: Pain
    - U: Unresponsive
    
    Args:
        x: Consciousness level value(s)
        
    Returns:
        AVPU level(s) as string(s)
        
    Examples:
        >>> avpu('Alert')
        'A'
        >>> avpu(['Alert', 'Unresponsive'])
        ['A', 'U']
    """
    if isinstance(x, pd.Series):
        return x.map(_avpu_map, na_action='ignore')
    
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_avpu_map(v) for v in x]
    
    return _avpu_map(x)


def _avpu_map(val) -> Optional[str]:
    """Map value to AVPU."""
    if pd.isna(val):
        return None
    
    val_str = str(val).upper()
    
    if 'ALERT' in val_str or val_str == 'A':
        return 'A'
    elif 'VOICE' in val_str or val_str == 'V':
        return 'V'
    elif 'PAIN' in val_str or val_str == 'P':
        return 'P'
    elif 'UNRESPONSIVE' in val_str or 'UNRESPONS' in val_str or val_str == 'U':
        return 'U'
    
    return None


def bmi(weight_kg: Union[float, pd.Series], height_m: Union[float, pd.Series]) -> Union[float, pd.Series]:
    """Calculate Body Mass Index (R ricu bmi).
    
    BMI = weight (kg) / height (m)^2
    
    Args:
        weight_kg: Weight in kilograms
        height_m: Height in meters
        
    Returns:
        BMI value(s)
        
    Examples:
        >>> bmi(70, 1.75)
        22.86
    """
    if isinstance(weight_kg, pd.Series) or isinstance(height_m, pd.Series):
        weight_kg = pd.Series(weight_kg) if not isinstance(weight_kg, pd.Series) else weight_kg
        height_m = pd.Series(height_m) if not isinstance(height_m, pd.Series) else height_m
        return weight_kg / (height_m ** 2)
    
    return weight_kg / (height_m ** 2)


def gcs(eye: Optional[Union[int, pd.Series]] = None,
        verbal: Optional[Union[int, pd.Series]] = None,
        motor: Optional[Union[int, pd.Series]] = None,
        total: Optional[Union[int, pd.Series]] = None) -> Union[int, pd.Series]:
    """Glasgow Coma Scale (R ricu gcs).
    
    GCS is calculated as sum of eye, verbal, and motor components,
    or can be provided directly as total.
    
    Args:
        eye: Eye opening score (1-4)
        verbal: Verbal response score (1-5)
        motor: Motor response score (1-6)
        total: Total GCS score (if provided, used directly)
        
    Returns:
        GCS score(s) (3-15)
        
    Examples:
        >>> gcs(eye=4, verbal=5, motor=6)
        15
        >>> gcs(total=15)
        15
    """
    if total is not None:
        if isinstance(total, pd.Series):
            return total
        return total
    
    if eye is None or verbal is None or motor is None:
        raise ValueError("Either total or all of eye, verbal, motor must be provided")
    
    if isinstance(eye, pd.Series) or isinstance(verbal, pd.Series) or isinstance(motor, pd.Series):
        eye = pd.Series(eye) if not isinstance(eye, pd.Series) else eye
        verbal = pd.Series(verbal) if not isinstance(verbal, pd.Series) else verbal
        motor = pd.Series(motor) if not isinstance(motor, pd.Series) else motor
        return eye + verbal + motor
    
    return eye + verbal + motor


def norepi_equiv(rate: Union[float, pd.Series], 
                 drug: Optional[Union[str, pd.Series]] = None) -> Union[float, pd.Series]:
    """Norepinephrine equivalent calculation (R ricu norepi_equiv).
    
    Converts vasopressor rates to norepinephrine equivalents.
    
    Args:
        rate: Infusion rate
        drug: Drug name (default: norepinephrine)
        
    Returns:
        Norepinephrine equivalent rate(s)
        
    Examples:
        >>> norepi_equiv(0.1, 'epinephrine')
        0.1
    """
    # Conversion factors (micrograms/kg/min)
    conversion_factors = {
        'norepinephrine': 1.0,
        'epinephrine': 1.0,
        'dopamine': 0.1,
        'phenylephrine': 0.45,
        'vasopressin': 0.01,  # Units conversion needed
    }
    
    if isinstance(rate, pd.Series):
        if drug is None or (isinstance(drug, pd.Series) and drug.isna().all()):
            return rate  # Assume norepinephrine
        
        drug = pd.Series(drug) if not isinstance(drug, pd.Series) else drug
        factors = drug.map(lambda d: conversion_factors.get(str(d).lower(), 1.0) if pd.notna(d) else 1.0)
        return rate * factors
    
    if drug is None:
        return rate
    
    factor = conversion_factors.get(str(drug).lower(), 1.0)
    return rate * factor


def supp_o2(o2_flow: Optional[Union[float, pd.Series]] = None,
            o2_device: Optional[Union[str, pd.Series]] = None) -> Union[bool, pd.Series]:
    """Supplemental oxygen indicator (R ricu supp_o2).
    
    Determines if patient is receiving supplemental oxygen.
    
    Args:
        o2_flow: Oxygen flow rate
        o2_device: Oxygen device type
        
    Returns:
        Boolean(s) indicating supplemental oxygen
        
    Examples:
        >>> supp_o2(o2_flow=5.0)
        True
        >>> supp_o2(o2_device='Nasal Cannula')
        True
    """
    if isinstance(o2_flow, pd.Series) or isinstance(o2_device, pd.Series):
        o2_flow = pd.Series(o2_flow) if not isinstance(o2_flow, pd.Series) else o2_flow
        o2_device = pd.Series(o2_device) if not isinstance(o2_device, pd.Series) else o2_device
        
        result = pd.Series(False, index=o2_flow.index if isinstance(o2_flow, pd.Series) else o2_device.index)
        
        if o2_flow is not None:
            result = result | (o2_flow > 0)
        
        if o2_device is not None:
            device_mask = o2_device.str.contains('Cannula|Mask|Venturi|High Flow|CPAP|BiPAP', 
                                                 case=False, na=False)
            result = result | device_mask
        
        return result
    
    result = False
    
    if o2_flow is not None and o2_flow > 0:
        result = True
    
    if o2_device is not None:
        device_str = str(o2_device).lower()
        if any(term in device_str for term in ['cannula', 'mask', 'venturi', 'high flow', 'cpap', 'bipap']):
            result = True
    
    return result


def urine24(urine_vol: Union[float, pd.Series],
            duration_hours: Optional[Union[float, pd.Series]] = None) -> Union[float, pd.Series]:
    """Calculate 24-hour urine output (R ricu urine24).
    
    Args:
        urine_vol: Urine volume in mL
        duration_hours: Duration in hours (default: 24)
        
    Returns:
        24-hour equivalent urine output
        
    Examples:
        >>> urine24(1000, duration_hours=12)
        2000.0
    """
    if duration_hours is None:
        duration_hours = 24.0
    
    if isinstance(urine_vol, pd.Series) or isinstance(duration_hours, pd.Series):
        urine_vol = pd.Series(urine_vol) if not isinstance(urine_vol, pd.Series) else urine_vol
        duration_hours = pd.Series(duration_hours) if not isinstance(duration_hours, pd.Series) else duration_hours
        return urine_vol * (24.0 / duration_hours)
    
    return urine_vol * (24.0 / duration_hours)


def vaso60(rate: Union[float, pd.Series]) -> Union[bool, pd.Series]:
    """Vasoactive drug at 60 minutes indicator (R ricu vaso60).
    
    Args:
        rate: Vasoactive drug rate
        
    Returns:
        Boolean(s) indicating vasoactive drug use
        
    Examples:
        >>> vaso60(0.1)
        True
    """
    if isinstance(rate, pd.Series):
        return rate > 0
    
    return rate > 0 if rate is not None else False


def vaso_ind(rate: Union[float, pd.Series]) -> Union[bool, pd.Series]:
    """Vasoactive drug indicator (R ricu vaso_ind).
    
    Args:
        rate: Vasoactive drug rate
        
    Returns:
        Boolean(s) indicating vasoactive drug use
        
    Examples:
        >>> vaso_ind(0.1)
        True
    """
    return vaso60(rate)  # Same logic


def vent_ind(vent_status: Union[bool, str, pd.Series]) -> Union[bool, pd.Series]:
    """Ventilation indicator (R ricu vent_ind).
    
    Args:
        vent_status: Ventilation status (boolean or string)
        
    Returns:
        Boolean(s) indicating ventilation
        
    Examples:
        >>> vent_ind('Mechanical Ventilation')
        True
        >>> vent_ind(True)
        True
    """
    if isinstance(vent_status, pd.Series):
        if pd.api.types.is_bool_dtype(vent_status):
            return vent_status
        return vent_status.str.contains('Vent|Mechanical|Invasive', case=False, na=False)
    
    if isinstance(vent_status, bool):
        return vent_status
    
    if isinstance(vent_status, str):
        return any(term in vent_status.lower() for term in ['vent', 'mechanical', 'invasive'])
    
    return bool(vent_status)

