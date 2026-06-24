"""Serialise the concept catalog into the shape the frontend expects.

The frontend (``static/js/screens-dict.js`` and friends) reads a global
``window.EU_CATALOG`` object:

    { groups, groupConcepts, dict, cov, desc, totalConcepts }

All of that data already exists, hand-curated for the UI, in
``easyicu.concept_catalog``. This module is the single source of
truth for the migration's first read-only endpoint — it just reshapes
those dicts; it does not recompute anything.
"""
from __future__ import annotations

from typing import Any, Dict

from easyicu import concept_catalog as cc


def build_catalog() -> Dict[str, Any]:
    # groups: ordered [key, name_en, name_zh] mirroring the curated order.
    groups = [
        [gk, *cc.CONCEPT_GROUP_NAMES.get(gk, (gk, gk))]
        for gk in cc.CONCEPT_GROUPS_INTERNAL
    ]
    group_concepts = {gk: list(members) for gk, members in cc.CONCEPT_GROUPS_INTERNAL.items()}

    # dict[k] = [name_en, name_zh, unit]; tuples -> lists for JSON.
    concept_dict = {k: list(v) for k, v in cc.CONCEPT_DICTIONARY.items()}
    desc = {k: list(v) for k, v in cc.CONCEPT_DESCRIPTIONS.items()}
    cov = dict(cc.CONCEPT_DB_COVERAGE)

    return {
        "groups": groups,
        "groupConcepts": group_concepts,
        "dict": concept_dict,
        "desc": desc,
        "cov": cov,
        "supportedDbs": list(cc.SUPPORTED_DB_KEYS),
        "totalConcepts": len(concept_dict),
    }
