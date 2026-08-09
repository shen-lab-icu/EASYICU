"""Convenience loaders for common ICU concept bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd

from .concepts import load_concepts
from .special_concepts import _validate_concepts


def load_sofa(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    keep_components: bool = True,
    verbose: bool = False,
    **kwargs,  # 允许传递额外参数如align_to_admission
) -> pd.DataFrame:
    """
    加载SOFA评分（便捷函数）- 重构版本

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        keep_components: 是否保留组件（默认True）
        verbose: 是否显示详细信息
        **kwargs: 额外参数传递给load_concepts（如align_to_admission）

    Returns:
        SOFA评分DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> sofa = load_sofa(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> sofa = load_sofa(patient_ids=[123, 456],
        ...                  database='miiv', data_path='/data/miiv',
        ...                  win_length='12h', interval='6h')
        >>>
        >>> # 使用时间对齐
        >>> sofa = load_sofa(patient_ids=[123, 456],
        ...                  align_to_admission=True)
    """
    if verbose:
        print("🏥 加载SOFA评分...")

    return load_concepts(
        "sofa",
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        keep_components=keep_components,
        verbose=verbose,
        **kwargs,  # 传递额外参数
    )


def load_sofa2(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    keep_components: bool = True,
    verbose: bool = False,
    **kwargs,  # 允许传递额外参数如align_to_admission
) -> pd.DataFrame:
    """
    加载SOFA-2评分（2025年新标准）- 重构版本

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        keep_components: 是否保留组件（默认True）
        verbose: 是否显示详细信息
        **kwargs: 额外参数传递给load_concepts（如align_to_admission）

    Returns:
        SOFA-2评分DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> sofa2 = load_sofa2(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> sofa2 = load_sofa2(patient_ids=[123, 456],
        ...                   database='miiv', data_path='/data/miiv')
    """
    if verbose:
        print("🏥 加载SOFA-2评分（2025标准）...")

    return load_concepts(
        "sofa2",
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        keep_components=keep_components,
        verbose=verbose,
        use_sofa2=True,  # 强制使用SOFA2字典
        **kwargs,  # 传递额外参数
    )


def load_sepsis3(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载Sepsis-3诊断相关数据 - 重构版本

    包含: SOFA, abx, samp, susp_inf, sep3

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        verbose: 是否显示详细信息

    Returns:
        Sepsis-3数据DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> sep3 = load_sepsis3(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> sep3 = load_sepsis3(patient_ids=[123, 456],
        ...                     database='miiv', data_path='/data/miiv')
    """
    if verbose:
        print("🦠 加载Sepsis-3相关数据...")

    # 只加载sep3概念，它已经包含了所有必需的诊断信息
    # 如果需要详细的组件（SOFA, abx等），用户可以分别加载
    return load_concepts(
        "sep3",
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        verbose=verbose,
    )


def load_vitals(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载生命体征数据（便捷函数）- 重构版本

    包含: hr, sbp, dbp, temp, resp, spo2

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        verbose: 是否显示详细信息

    Returns:
        生命体征DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> vitals = load_vitals(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> vitals = load_vitals(patient_ids=[123, 456],
        ...                      database='miiv', data_path='/data/miiv',
        ...                      interval='30m')
    """
    vital_concepts = ["hr", "sbp", "dbp", "temp", "resp", "spo2"]

    if verbose:
        print("❤️  加载生命体征...")

    return load_concepts(
        vital_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        verbose=verbose,
    )


def load_labs(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "6h",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载实验室检查数据（便捷函数）- 重构版本

    包含: wbc, plt, crea, bili, lact, ph

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认6小时，实验室检查频率较低）
        verbose: 是否显示详细信息

    Returns:
        实验室检查DataFrame

    Examples:
        >>> # 最简单的用法 - 自动检测
        >>> labs = load_labs(patient_ids=[123, 456])
        >>>
        >>> # 完全自定义
        >>> labs = load_labs(patient_ids=[123, 456],
        ...                   database='miiv', data_path='/data/miiv',
        ...                   interval='12h')
    """
    lab_concepts = ["wbc", "plt", "crea", "bili", "lact", "ph"]

    if verbose:
        print("🔬 加载实验室检查...")

    return load_concepts(
        lab_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        verbose=verbose,
    )


def load_demographics(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载基础人口统计学数据（参考ricu.R的data_demo）

    包含: age, bmi, height, sex, weight

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        人口统计学DataFrame

    Examples:
        >>> demo = load_demographics(patient_ids=[123, 456])
    """
    if verbose:
        print("👥 加载基础人口统计学数据...")

    demo_concepts = ["age", "bmi", "height", "sex", "weight"]

    try:
        result = load_concepts(
            concepts=demo_concepts,
            patient_ids=patient_ids,
            database=database,
            data_path=data_path,
            merge=True,
            verbose=verbose,
        )
        if result is None:
            return pd.DataFrame()
        return result

    except Exception as e:
        if verbose:
            print(f"  ❌ 人口统计学数据加载失败: {e}")
        return pd.DataFrame()


def load_outcomes(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    keep_components: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载结局指标数据（参考ricu.R的data_outcome）

    包含: death, los_icu, qsofa, sirs

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        keep_components: 是否保留组件（默认True）
        verbose: 是否显示详细信息

    Returns:
        结局指标DataFrame

    Examples:
        >>> outcomes = load_outcomes(patient_ids=[123, 456])
    """
    if verbose:
        print("📊 加载结局指标数据...")

    concepts = ["death", "los_icu", "qsofa", "sirs"]
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        keep_components=keep_components,
        merge=True,
        verbose=verbose,
    )


def load_vitals_detailed(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载详细生命体征数据（参考ricu.R的data_vital）

    包含: dbp, etco2, hr, map, sbp, temp

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        详细生命体征DataFrame

    Examples:
        >>> vitals = load_vitals_detailed(patient_ids=[123, 456])
    """
    if verbose:
        print("❤️ 加载详细生命体征数据...")

    concepts = ["dbp", "etco2", "hr", "map", "sbp", "temp"]
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose,
    )


def load_neurological(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载神经系统评估数据（参考ricu.R的data_neu）

    包含: avpu, egcs, gcs, mgcs, rass, vgcs

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        神经系统评估DataFrame

    Examples:
        >>> neuro = load_neurological(patient_ids=[123, 456])
    """
    if verbose:
        print("🧠 加载神经系统评估数据...")

    concepts = ["avpu", "egcs", "gcs", "mgcs", "rass", "vgcs"]
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose,
    )


def load_output(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载输出量数据（参考ricu.R的data_output）

    包含: urine, urine24, total_input_ml, fluid_balance, fluid_balance_cumulative

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        输出量DataFrame

    Examples:
        >>> output = load_output(patient_ids=[123, 456])
    """
    if verbose:
        print("💧 加载输出量数据...")

    concepts = [
        "urine",
        "urine24",
        "total_input_ml",
        "fluid_balance",
        "fluid_balance_cumulative",
    ]
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose,
    )


def load_respiratory(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载呼吸系统数据（参考ricu.R的data_resp）

    包含: ett_gcs, mech_vent, o2sat, sao2, pafi, resp, safi, supp_o2, vent_ind

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        呼吸系统DataFrame

    Examples:
        >>> resp = load_respiratory(patient_ids=[123, 456])
    """
    if verbose:
        print("🫁 加载呼吸系统数据...")

    concepts = [
        "ett_gcs",
        "mech_vent",
        "o2sat",
        "sao2",
        "pafi",
        "resp",
        "safi",
        "supp_o2",
        "vent_ind",
    ]
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose,
    )


def load_lab_comprehensive(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载全面的实验室检查数据（参考ricu.R的data_lab）

    包含: alb, alp, alt, ast, bicar, bili, bili_dir, bun, ca, ck, ckmb,
          cl, crea, crp, glu, k, mg, na, phos, tnt

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        实验室检查DataFrame

    Examples:
        >>> labs = load_lab_comprehensive(patient_ids=[123, 456])
    """
    if verbose:
        print("🧪 加载全面的实验室检查数据...")

    concepts = [
        "alb",
        "alp",
        "alt",
        "ast",
        "bicar",
        "bili",
        "bili_dir",
        "bun",
        "ca",
        "ck",
        "ckmb",
        "cl",
        "crea",
        "crp",
        "glu",
        "k",
        "mg",
        "na",
        "phos",
        "tnt",
    ]
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose,
    )


def load_blood_gas(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载血气分析数据（参考ricu.R的data_blood）

    包含: be, cai, fio2, hbco, lact, methb, pco2, ph, po2, tco2

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        血气分析DataFrame

    Examples:
        >>> blood_gas = load_blood_gas(patient_ids=[123, 456])
    """
    if verbose:
        print("🩸 加载血气分析数据...")

    concepts = [
        "be",
        "cai",
        "fio2",
        "hbco",
        "lact",
        "methb",
        "pco2",
        "ph",
        "po2",
        "tco2",
    ]
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    # 逐个尝试加载，跳过无法加载的概念（某些概念可能在特定数据库中没有配置）
    results = []
    loaded_concepts = []
    for concept in available_concepts:
        try:
            df = load_concepts(
                concepts=[concept],
                patient_ids=patient_ids,
                database=database,
                data_path=data_path,
                interval=interval,
                win_length=win_length,
                merge=True,
                verbose=False,
            )
            if df is not None and not df.empty:
                results.append(df)
                loaded_concepts.append(concept)
        except Exception:
            pass  # 跳过无法加载的概念

    if not results:
        if verbose:
            print("  ❌ 没有成功加载的概念")
        return pd.DataFrame()

    if verbose:
        print(f"  ✅ 成功加载 {len(loaded_concepts)} 个概念: {loaded_concepts}")

    # 合并结果
    if len(results) == 1:
        return results[0]

    # 多个结果需要合并
    merged = results[0]
    for df in results[1:]:
        # 找到共同的 ID 和时间列进行合并
        id_cols = [
            c
            for c in merged.columns
            if "id" in c.lower()
            or c
            in [
                "stay_id",
                "subject_id",
                "patientunitstayid",
                "admissionid",
                "patientid",
            ]
        ]
        time_cols = [
            c for c in merged.columns if "time" in c.lower() or c == "charttime"
        ]
        merge_cols = list(set(id_cols + time_cols) & set(df.columns))
        if merge_cols:
            merged = pd.merge(merged, df, on=merge_cols, how="outer")
        else:
            merged = pd.concat([merged, df], ignore_index=True)

    return merged


def load_hematology(
    patient_ids: Optional[Union[List, Dict]] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: Union[str, pd.Timedelta] = "1h",
    win_length: Union[str, pd.Timedelta] = "24h",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    加载血液学检查数据（参考ricu.R的data_hematology）

    包含: bnd, esr, fgn, hgb, inr_pt, lymph, mch, mchc, mcv, neut, plt, ptt, wbc

    Args:
        patient_ids: 患者ID列表（None=所有患者）
        database: 数据库类型 (None=自动检测)
        data_path: 数据路径 (None=自动检测)
        interval: 时间间隔（默认1小时）
        win_length: 窗口长度（默认24小时）
        verbose: 是否显示详细信息

    Returns:
        血液学DataFrame

    Examples:
        >>> hematology = load_hematology(patient_ids=[123, 456])
    """
    if verbose:
        print("🩸 加载血液学检查数据...")

    concepts = [
        "bnd",
        "esr",
        "fgn",
        "hgb",
        "inr_pt",
        "lymph",
        "mch",
        "mchc",
        "mcv",
        "neut",
        "plt",
        "ptt",
        "wbc",
    ]
    available_concepts = _validate_concepts(concepts, verbose)

    if not available_concepts:
        if verbose:
            print("  ❌ 没有可用的概念")
        return pd.DataFrame()

    return load_concepts(
        concepts=available_concepts,
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        win_length=win_length,
        merge=True,
        verbose=verbose,
    )


def load_sofa_with_score_impl(
    *,
    load_concepts_fn: Callable[..., pd.DataFrame],
    patient_ids: Optional[List] = None,
    database: Optional[str] = None,
    data_path: Optional[Union[str, Path]] = None,
    interval: str = "1h",
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Load SOFA total and component scores through the concept service."""
    return load_concepts_fn(
        concepts=[
            "sofa",
            "sofa_resp",
            "sofa_coag",
            "sofa_liver",
            "sofa_cardio",
            "sofa_cns",
            "sofa_renal",
        ],
        patient_ids=patient_ids,
        database=database,
        data_path=data_path,
        interval=interval,
        merge=True,
        verbose=verbose,
        **kwargs,
    )


__all__ = [
    "load_sofa",
    "load_sofa2",
    "load_sepsis3",
    "load_vitals",
    "load_labs",
    "load_demographics",
    "load_outcomes",
    "load_vitals_detailed",
    "load_neurological",
    "load_output",
    "load_respiratory",
    "load_lab_comprehensive",
    "load_blood_gas",
    "load_hematology",
    "load_sofa_with_score_impl",
]
