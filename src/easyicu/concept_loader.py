"""Concept-dictionary loading shims.

Extracted from :mod:`easyicu.concept` (2026-05-17) as part of the
Phase-1 split documented in CLAUDE.md. These three small helpers
formerly lived interleaved with the resolver code (former lines
~7934-7950 + ~10637-10651).

* :func:`_load_concept_dict_cached` — read ``data/concept-dict.json``
  once and cache the raw dict in module memory.
* :func:`_get_concept_bounds` — pull a ``min`` or ``max`` value out of
  the cached raw dict; used by ``filter_bounds`` plumbing inside the
  resolver and by :mod:`easyicu.api`.
* :func:`load_dictionary` — public compatibility wrapper that delegates
  to :func:`easyicu.resources.load_dictionary`; the ``src_name`` argument
  is accepted for backward compatibility but currently ignored.

Public surface
--------------
All three names are re-exported by :mod:`easyicu.concept`. Existing
``from easyicu.concept import load_dictionary`` keeps working.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Optional

from .concept_schema import ConceptDictionary


@functools.lru_cache(maxsize=1)
def _load_concept_dict_cached():
    """Load concept-dict.json once and cache in memory."""
    dict_path = Path(__file__).parent / 'data' / 'concept-dict.json'
    with open(dict_path) as f:
        return json.load(f)


def _get_concept_bounds(concept_name: str, bound: str) -> Optional[float]:
    """Get min/max bounds from concept-dict.json for filter_bounds."""
    try:
        d = _load_concept_dict_cached()
        c = d.get(concept_name, {})
        val = c.get(bound)
        return float(val) if val is not None else None
    except Exception:
        return None


def load_dictionary(src_name: Optional[str] = None, include_sofa2: bool = False) -> ConceptDictionary:
    """加载概念字典 - 兼容函数

    Args:
        src_name: 数据源名称（可选，当前未使用，保留以兼容旧调用）
        include_sofa2: 是否包含 SOFA-2 概念字典

    Returns:
        ConceptDictionary 实例
    """
    # Lazy import to avoid a circular dependency on easyicu.resources at
    # module load time (resources -> concept_loader -> resources ...).
    from .resources import load_dictionary as _load_dictionary

    # 当前实现不根据数据源过滤概念，但保留参数以兼容既有调用
    return _load_dictionary(include_sofa2=include_sofa2)


__all__ = [
    "_load_concept_dict_cached",
    "_get_concept_bounds",
    "load_dictionary",
]
