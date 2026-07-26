"""Special-concept catalog inspection and validation services."""

from __future__ import annotations

from typing import Dict, List, Optional

from ..config import load_data_sources as load_user_data_sources
from ..resources import load_data_sources as load_packaged_data_sources
from ..resources import load_dictionary


def list_available_concepts(source: Optional[str] = None) -> List[str]:
    """
    列出可用的概念

    Args:
        source: 如果指定，只列出该数据源支持的概念

    Returns:
        概念名称列表

    Examples:
        >>> # 列出所有概念
        >>> all_concepts = list_available_concepts()
        >>>
        >>> # 列出MIMIC支持的概念
        >>> mimic_concepts = list_available_concepts('mimic')
    """
    dict_obj = load_dictionary()

    if source is None:
        # 返回所有概念 (使用 _concepts 属性)
        return list(dict_obj._concepts.keys())

    # 返回特定数据源支持的概念
    supported = []
    for name, concept in dict_obj._concepts.items():
        if hasattr(concept, "sources") and source in concept.sources:
            supported.append(name)

    return sorted(supported)


def list_available_sources(use_user_config: bool = False) -> List[str]:
    """
    列出可用的数据源

    Args:
        use_user_config: If True, read the legacy user configuration registry.
            By default this reports packaged sources shipped with EasyICU.

    Returns:
        数据源名称列表

    Examples:
        >>> sources = list_available_sources()
        >>> print(sources)
        ['mimic', 'hirid', 'eicu', 'aumc']
    """
    registry = (
        load_user_data_sources() if use_user_config else load_packaged_data_sources()
    )
    return [cfg.name for cfg in registry]


def get_concept_info(concept_name: str) -> Dict:
    """
    获取概念的详细信息

    Args:
        concept_name: 概念名称

    Returns:
        包含概念信息的字典

    Examples:
        >>> info = get_concept_info('hr')
        >>> print(info['description'])
        'heart rate'
    """
    dict_obj = load_dictionary()
    concept = dict_obj.get(concept_name)

    if concept is None:
        raise ValueError(f"未知概念: {concept_name}")

    units = list(getattr(concept, "units", None) or [])
    sources = getattr(concept, "sources", {}) or {}

    info = {
        "name": concept_name,
        "description": getattr(concept, "description", ""),
        "category": getattr(concept, "category", ""),
        "units": units,
        "unit": units[0] if units else "",
        "sources": sorted(sources.keys()),
        "class_name": getattr(concept, "class_name", None),
        "callback": getattr(concept, "callback", None),
        "sub_concepts": list(getattr(concept, "sub_concepts", None) or []),
        "depends_on": list(getattr(concept, "depends_on", None) or []),
    }

    return info


def _validate_concepts(concepts: List[str], verbose: bool = False) -> List[str]:
    """
    验证概念是否存在于字典中，返回可用的概念列表

    Args:
        concepts: 要验证的概念列表
        verbose: 是否显示详细信息

    Returns:
        可用的概念列表
    """
    try:
        dict_obj = load_dictionary()
        # 使用 _concepts 属性 (ConceptDictionary 内部存储)
        all_concepts = set(dict_obj._concepts.keys())
        available_concepts = [c for c in concepts if c in all_concepts]
        missing_concepts = [c for c in concepts if c not in all_concepts]

        if verbose and missing_concepts:
            print(f"  ⚠️  以下概念在字典中不存在，将被跳过: {missing_concepts}")

        return available_concepts
    except Exception:
        return concepts  # 如果验证失败，返回原列表


__all__ = [
    "list_available_concepts",
    "list_available_sources",
    "get_concept_info",
    "_validate_concepts",
]
