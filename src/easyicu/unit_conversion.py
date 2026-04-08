"""
单位转换系统
实现医学常用单位之间的转换 (完全复刻 R ricu callback-itm.R 的 convert_unit 功能)
"""
from typing import Optional, Union
import pandas as pd
import numpy as np

class UnitConverter:
    """医学单位转换器 - 完整实现 R ricu 的单位转换功能"""
    
    # 单位转换系数表 (扩展自 R ricu)
    CONVERSIONS = {
        # 温度转换 (Fahrenheit <-> Celsius)
        ('celsius', 'fahrenheit'): lambda x: x * 9/5 + 32,
        ('fahrenheit', 'celsius'): lambda x: (x - 32) * 5/9,
        ('celsius', 'kelvin'): lambda x: x + 273.15,
        ('kelvin', 'celsius'): lambda x: x - 273.15,
        ('f', 'c'): lambda x: (x - 32) * 5/9,
        ('c', 'f'): lambda x: x * 9/5 + 32,
        
        # 葡萄糖: mg/dL <-> mmol/L (molecular weight: 180.16)
        ('mg/dl', 'mmol/l'): lambda x: x / 18.016,
        ('mmol/l', 'mg/dl'): lambda x: x * 18.016,
        
        # 肌酐: mg/dL <-> µmol/L
        ('mg/dl', 'umol/l'): lambda x: x * 88.4,
        ('umol/l', 'mg/dl'): lambda x: x / 88.4,
        
        # 尿素: mg/dL <-> mmol/L
        ('mg/dl', 'mmol/l', 'urea'): lambda x: x / 2.8,
        ('mmol/l', 'mg/dl', 'urea'): lambda x: x * 2.8,
        
        # 乳酸: mg/dL <-> mmol/L
        ('mg/dl', 'mmol/l', 'lactate'): lambda x: x / 9.0,
        ('mmol/l', 'mg/dl', 'lactate'): lambda x: x * 9.0,
        
        # 胆红素: mg/dL <-> µmol/L
        ('mg/dl', 'umol/l', 'bilirubin'): lambda x: x * 17.1,
        ('umol/l', 'mg/dl', 'bilirubin'): lambda x: x / 17.1,
        
        # 血红蛋白: g/dL <-> g/L
        ('g/dl', 'g/l'): lambda x: x * 10,
        ('g/l', 'g/dl'): lambda x: x / 10,
        
        # 白细胞: K/µL <-> 10^9/L
        ('k/ul', '10^9/l'): lambda x: x,  # 等价
        ('10^9/l', 'k/ul'): lambda x: x,
        
        # 血小板: K/µL <-> 10^9/L
        ('k/ul', '10^9/l', 'platelet'): lambda x: x,
        ('10^9/l', 'k/ul', 'platelet'): lambda x: x,
        
        # 压力: mmHg <-> kPa
        ('mmhg', 'kpa'): lambda x: x * 0.133322,
        ('kpa', 'mmhg'): lambda x: x / 0.133322,
        
        # 体重: kg <-> lb
        ('kg', 'lb'): lambda x: x * 2.20462,
        ('lb', 'kg'): lambda x: x / 2.20462,
        
        # 身高: cm <-> inch
        ('cm', 'inch'): lambda x: x / 2.54,
        ('inch', 'cm'): lambda x: x * 2.54,
        
        # 体积: mL <-> L
        ('ml', 'l'): lambda x: x / 1000,
        ('l', 'ml'): lambda x: x * 1000,
        
        # FiO2: % <-> fraction
        ('%', 'fraction', 'fio2'): lambda x: x / 100,
        ('fraction', '%', 'fio2'): lambda x: x * 100,
        
        # Vasopressor rate conversions (CRITICAL for SOFA cardiovascular)
        # μg/kg/min is the standard unit for SOFA cardiovascular scoring
        # Source units may vary: μg/min (not weight-adjusted), mg/h, etc.
        # Note: These require patient weight (kg) for conversion
        # Use convert_vaso_rate() helper function instead of direct conversion
        
        # 药物剂量常用转换
        ('mg', 'g'): lambda x: x / 1000,
        ('g', 'mg'): lambda x: x * 1000,
        ('ug', 'mg'): lambda x: x / 1000,
        ('mg', 'ug'): lambda x: x * 1000,
        ('ng', 'ug'): lambda x: x / 1000,
        ('ug', 'ng'): lambda x: x * 1000,
        ('ng', 'mg'): lambda x: x / 1000000,
        ('mg', 'ng'): lambda x: x * 1000000,
        
        # 时间转换
        ('sec', 'min'): lambda x: x / 60,
        ('min', 'sec'): lambda x: x * 60,
        ('min', 'hour'): lambda x: x / 60,
        ('hour', 'min'): lambda x: x * 60,
        ('hour', 'day'): lambda x: x / 24,
        ('day', 'hour'): lambda x: x * 24,
        ('day', 'week'): lambda x: x / 7,
        ('week', 'day'): lambda x: x * 7,
        
        # 尿素氮 (BUN): mg/dL <-> mmol/L
        ('mg/dl', 'mmol/l', 'bun'): lambda x: x / 2.8,
        ('mmol/l', 'mg/dl', 'bun'): lambda x: x * 2.8,
        
        # 蛋白质: g/dL <-> g/L
        ('g/dl', 'g/l', 'protein'): lambda x: x * 10,
        ('g/l', 'g/dl', 'protein'): lambda x: x / 10,
        
        # 钙: mg/dL <-> mmol/L
        ('mg/dl', 'mmol/l', 'calcium'): lambda x: x / 4.0,
        ('mmol/l', 'mg/dl', 'calcium'): lambda x: x * 4.0,
        
        # 镁: mg/dL <-> mmol/L (或 mEq/L)
        ('mg/dl', 'mmol/l', 'magnesium'): lambda x: x / 2.4,
        ('mmol/l', 'mg/dl', 'magnesium'): lambda x: x * 2.4,
        ('mg/dl', 'meq/l', 'magnesium'): lambda x: x / 1.2,
        ('meq/l', 'mg/dl', 'magnesium'): lambda x: x * 1.2,
        
        # 磷: mg/dL <-> mmol/L
        ('mg/dl', 'mmol/l', 'phosphate'): lambda x: x / 3.1,
        ('mmol/l', 'mg/dl', 'phosphate'): lambda x: x * 3.1,
        
        # 胆固醇: mg/dL <-> mmol/L
        ('mg/dl', 'mmol/l', 'cholesterol'): lambda x: x / 38.67,
        ('mmol/l', 'mg/dl', 'cholesterol'): lambda x: x * 38.67,
        
        # 甘油三酯: mg/dL <-> mmol/L
        ('mg/dl', 'mmol/l', 'triglyceride'): lambda x: x / 88.57,
        ('mmol/l', 'mg/dl', 'triglyceride'): lambda x: x * 88.57,
        
        # 流速: mL/min <-> L/min
        ('ml/min', 'l/min'): lambda x: x / 1000,
        ('l/min', 'ml/min'): lambda x: x * 1000,
        
        # 流速: mL/h <-> mL/min
        ('ml/h', 'ml/min'): lambda x: x / 60,
        ('ml/min', 'ml/h'): lambda x: x * 60,
        
        # 能量: kcal <-> kJ
        ('kcal', 'kj'): lambda x: x * 4.184,
        ('kj', 'kcal'): lambda x: x / 4.184,
    }
    
    # 单位别名
    UNIT_ALIASES = {
        'fahrenheit': ['f', 'deg f', '°f'],
        'celsius': ['c', 'deg c', '°c', 'centigrade'],
        'kelvin': ['k'],
        'mg/dl': ['mg/100ml', 'mg%', 'mg per dl'],
        'mmol/l': ['mmol/dl', 'mm'],
        'umol/l': ['µmol/l', 'micromol/l'],
        'g/dl': ['g/100ml', 'g%'],
        'g/l': ['grams/l'],
        'k/ul': ['k/µl', 'thou/ul', 'x10^3/µl'],
        '10^9/l': ['x10^9/l', 'giga/l'],
        'mmhg': ['mm hg', 'torr'],
        'kpa': ['kilopascal'],
        'kg': ['kilogram'],
        'lb': ['lbs', 'pound', 'pounds'],
        'cm': ['centimeter', 'centimeters'],
        'inch': ['in', 'inches', '"'],
        'ml': ['milliliter', 'milliliters'],
        'l': ['liter', 'liters'],
        '%': ['percent', 'pct'],
        'fraction': ['decimal'],
        'mg': ['milligram', 'milligrams'],
        'g': ['gram', 'grams'],
        'ug': ['µg', 'microgram', 'micrograms'],
        'min': ['minute', 'minutes'],
        'hour': ['hr', 'hours', 'hrs'],
        'day': ['days', 'd'],
        'sec': ['second', 'seconds', 's'],
        'week': ['weeks', 'wk', 'wks'],
        'ng': ['nanogram', 'nanograms'],
        'meq/l': ['meq/dl', 'milliequivalent/l'],
        'kcal': ['kilocalorie', 'kilocalories', 'cal'],
        'kj': ['kilojoule', 'kilojoules'],
    }
    
    @classmethod
    def normalize_unit(cls, unit: str) -> str:
        """
        标准化单位名称
        
        Args:
            unit: 原始单位字符串
            
        Returns:
            标准化的单位名称
        """
        unit_lower = unit.lower().strip()
        
        # 检查是否已经是标准名称
        if unit_lower in cls.UNIT_ALIASES:
            return unit_lower
        
        # 查找别名
        for standard, aliases in cls.UNIT_ALIASES.items():
            if unit_lower in aliases:
                return standard
        
        return unit_lower
    
    @property
    def conversions(self) -> dict:
        """访问类级别的 CONVERSIONS 字典 - 用于测试"""
        return self.CONVERSIONS
    
    @classmethod
    def can_convert(cls, from_unit: str, to_unit: str, 
                    substance: Optional[str] = None) -> bool:
        """
        检查是否可以转换
        
        Args:
            from_unit: 源单位
            to_unit: 目标单位
            substance: 物质名称（某些转换需要）
            
        Returns:
            是否可转换
        """
        from_norm = cls.normalize_unit(from_unit)
        to_norm = cls.normalize_unit(to_unit)
        
        if from_norm == to_norm:
            return True
        
        # 不带物质名的转换
        if (from_norm, to_norm) in cls.CONVERSIONS:
            return True
        
        # 带物质名的转换
        if substance and (from_norm, to_norm, substance.lower()) in cls.CONVERSIONS:
            return True
        
        return False
    
    @classmethod
    def convert(cls, value: Union[float, np.ndarray], 
                from_unit: str, to_unit: str,
                substance: Optional[str] = None) -> Union[float, np.ndarray]:
        """
        转换数值
        
        Args:
            value: 要转换的值
            from_unit: 源单位
            to_unit: 目标单位
            substance: 物质名称
            
        Returns:
            转换后的值
            
        Raises:
            ValueError: 如果无法转换
            
        Examples:
            >>> UnitConverter.convert(37.0, 'celsius', 'fahrenheit')
            98.6
            >>> UnitConverter.convert(100, 'mg/dl', 'mmol/l', substance='glucose')
            5.56
        """
        from_norm = cls.normalize_unit(from_unit)
        to_norm = cls.normalize_unit(to_unit)
        
        # 相同单位
        if from_norm == to_norm:
            return value
        
        # 查找转换函数
        converter = None
        
        # 尝试不带物质名
        if (from_norm, to_norm) in cls.CONVERSIONS:
            converter = cls.CONVERSIONS[(from_norm, to_norm)]
        # 尝试带物质名
        elif substance and (from_norm, to_norm, substance.lower()) in cls.CONVERSIONS:
            converter = cls.CONVERSIONS[(from_norm, to_norm, substance.lower())]
        
        if converter is None:
            raise ValueError(
                f"无法从 {from_unit} 转换到 {to_unit}" +
                (f" (物质: {substance})" if substance else "")
            )
        
        return converter(value)
    
    @classmethod
    def convert_series(cls, series: 'pd.Series', from_unit: str, to_unit: str,
                       substance: Optional[str] = None) -> 'pd.Series':
        """
        转换 Pandas Series
        
        Args:
            series: Pandas Series
            from_unit: 源单位
            to_unit: 目标单位
            substance: 物质名称
            
        Returns:
            转换后的 Series
        """
        import pandas as pd
        
        if series.empty:
            return series
        
        converted = cls.convert(series.values, from_unit, to_unit, substance)
        return pd.Series(converted, index=series.index, name=series.name)

def convert_unit(value: Union[float, np.ndarray, 'pd.Series'],
                 from_unit: str, to_unit: str,
                 substance: Optional[str] = None) -> Union[float, np.ndarray, 'pd.Series']:
    """
    便捷函数：单位转换
    
    Args:
        value: 要转换的值（标量、数组或Series）
        from_unit: 源单位
        to_unit: 目标单位
        substance: 物质名称（某些转换需要）
        
    Returns:
        转换后的值
        
    Examples:
        >>> # 温度转换
        >>> convert_unit(37.0, 'celsius', 'fahrenheit')
        98.6
        
        >>> # 葡萄糖转换
        >>> convert_unit(100, 'mg/dl', 'mmol/l')
        5.56
        
        >>> # 带物质名的转换
        >>> convert_unit(2.0, 'mg/dl', 'mmol/l', substance='lactate')
        0.22
    """
    # 检查是否是 Pandas Series
    try:
        import pandas as pd
        if isinstance(value, pd.Series):
            return UnitConverter.convert_series(value, from_unit, to_unit, substance)
    except ImportError:
        pass
    
    return UnitConverter.convert(value, from_unit, to_unit, substance)

# 预定义的常用转换函数
def celsius_to_fahrenheit(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """摄氏度转华氏度"""
    return temp * 9/5 + 32

def fahrenheit_to_celsius(temp: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """华氏度转摄氏度"""
    return (temp - 32) * 5/9

def glucose_mg_to_mmol(glucose: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """葡萄糖 mg/dL -> mmol/L"""
    return glucose / 18.0

def glucose_mmol_to_mg(glucose: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """葡萄糖 mmol/L -> mg/dL"""
    return glucose * 18.0

def creatinine_mg_to_umol(creat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """肌酐 mg/dL -> µmol/L"""
    return creat * 88.4

def creatinine_umol_to_mg(creat: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """肌酐 µmol/L -> mg/dL"""
    return creat / 88.4

def lactate_mg_to_mmol(lactate: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """乳酸 mg/dL -> mmol/L"""
    return lactate / 9.0

def lactate_mmol_to_mg(lactate: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """乳酸 mmol/L -> mg/dL"""
    return lactate * 9.0

def mmhg_to_kpa(pressure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """压力 mmHg -> kPa"""
    return pressure * 0.133322

def kpa_to_mmhg(pressure: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """压力 kPa -> mmHg"""
    return pressure / 0.133322

def convert_vaso_rate(
    rate: Union[float, np.ndarray, 'pd.Series'],
    from_unit: str,
    weight_kg: Optional[Union[float, np.ndarray, 'pd.Series']] = None,
) -> Union[float, np.ndarray, 'pd.Series']:
    """Convert vasopressor/inotrope infusion rates to μg/kg/min (SOFA standard).
    
    CRITICAL for SOFA cardiovascular scoring accuracy.
    
    Supported conversions:
    - 'ug/kg/min' or 'mcg/kg/min' → no conversion (already standard)
    - 'ug/min' or 'mcg/min' → divide by weight_kg
    - 'mg/h' or 'mg/hr' → (rate * 1000 μg/mg) / (60 min/h * weight_kg)
    - 'mg/kg/h' → (rate * 1000) / 60
    
    Args:
        rate: Infusion rate value(s)
        from_unit: Source unit string (case-insensitive)
        weight_kg: Patient weight in kg (required for non-weight-adjusted units)
        
    Returns:
        Rate in μg/kg/min
        
    Raises:
        ValueError: If weight is required but not provided, or unit is unsupported
        
    Examples:
        >>> # Already in standard unit
        >>> convert_vaso_rate(5.0, 'ug/kg/min')
        5.0
        
        >>> # Not weight-adjusted, need patient weight
        >>> convert_vaso_rate(300.0, 'ug/min', weight_kg=75.0)
        4.0
        
        >>> # From mg/h (common in some EHRs)
        >>> convert_vaso_rate(18.0, 'mg/h', weight_kg=60.0)
        5.0  # (18 * 1000) / (60 * 60) = 5.0
        
        >>> # Pandas Series
        >>> import pandas as pd
        >>> rates = pd.Series([300, 450, 600])
        >>> weights = pd.Series([75, 75, 75])
        >>> convert_vaso_rate(rates, 'ug/min', weight_kg=weights)
        0    4.0
        1    6.0
        2    8.0
        dtype: float64
    """
    # Normalize unit string
    unit = from_unit.lower().strip().replace(' ', '')
    
    # Handle pandas Series
    try:
        import pandas as pd
        is_series = isinstance(rate, pd.Series)
    except ImportError:
        is_series = False
    
    # Already in standard unit
    if unit in ['ug/kg/min', 'mcg/kg/min', 'μg/kg/min']:
        return rate
    
    # Conversions requiring weight
    if unit in ['ug/min', 'mcg/min', 'μg/min']:
        if weight_kg is None:
            raise ValueError(f"Patient weight (kg) required to convert from '{from_unit}' to μg/kg/min")
        if is_series:
            return rate / weight_kg
        else:
            return np.array(rate) / np.array(weight_kg) if isinstance(rate, (list, np.ndarray)) else rate / weight_kg
    
    if unit in ['mg/h', 'mg/hr']:
        if weight_kg is None:
            raise ValueError(f"Patient weight (kg) required to convert from '{from_unit}' to μg/kg/min")
        # mg/h → μg/kg/min: (mg/h * 1000 μg/mg) / (60 min/h * weight_kg)
        if is_series:
            return (rate * 1000.0) / (60.0 * weight_kg)
        else:
            rate_arr = np.array(rate) if isinstance(rate, (list, np.ndarray)) else rate
            weight_arr = np.array(weight_kg) if isinstance(weight_kg, (list, np.ndarray)) else weight_kg
            return (rate_arr * 1000.0) / (60.0 * weight_arr)
    
    if unit in ['mg/kg/h', 'mg/kg/hr']:
        # mg/kg/h → μg/kg/min: (mg/kg/h * 1000 μg/mg) / 60 min/h
        return rate * (1000.0 / 60.0)
    
    raise ValueError(
        f"Unsupported vasopressor rate unit: '{from_unit}'. "
        f"Supported: 'ug/kg/min', 'ug/min', 'mg/h', 'mg/kg/h'"
    )

