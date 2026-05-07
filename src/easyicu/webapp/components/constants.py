"""Compatibility exports for EasyICU web application constants."""

import streamlit as st

from ..concept_catalog import (
    CONCEPT_GROUP_NAMES,
    CONCEPT_GROUPS_DISPLAY,
    CONCEPT_GROUPS_INTERNAL,
)


def get_concept_groups():
    """根据当前语言返回带正确显示名称的特征分组。"""
    lang = st.session_state.get('language', 'en')
    groups = {}
    for key, concepts in CONCEPT_GROUPS_INTERNAL.items():
        if key in CONCEPT_GROUP_NAMES:
            en_name, cn_name = CONCEPT_GROUP_NAMES[key]
            display_name = en_name if lang == 'en' else cn_name
        else:
            display_name = key.replace('_', ' ').title()
        groups[display_name] = concepts
    return groups


# 获取所有可用概念的列表
def get_all_concepts():
    """获取所有可用概念的扁平列表。"""
    all_concepts = set()
    for group_concepts in CONCEPT_GROUPS_INTERNAL.values():
        all_concepts.update(group_concepts)
    return sorted(list(all_concepts))
