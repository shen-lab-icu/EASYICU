"""Display names for concept modules on the review surfaces.

Three implementations of the same lookup had drifted apart, and the drift was
visible to users: Cohort Statistics title-cased the raw module id without
consulting the concept catalog, so ``sofa2_score`` rendered as "Sofa2 Score"
and ``sepsis3_sofa2`` as "Sepsis3 Sofa2" — mangled clinical proper nouns —
while Patient Review showed the catalog's "SOFA-2 Scores" and "Sepsis-3
(SOFA-2 based)". Fourteen of the nineteen modules were named differently
depending on which page you were on, and Cohort Statistics filled its ``_zh``
field with the English string, so the Chinese UI showed it too.

The catalog is the source of truth for these names. This module is the only
thing that reads ``CONCEPT_GROUP_NAMES`` for display purposes; review surfaces
call it rather than re-deriving a label from the id.
"""

from __future__ import annotations

from typing import Dict

from easyicu.concept import catalog as concept_catalog


def plain_label(value: object) -> str:
    """Drop the catalog's decorative prefix so a name starts at its first word.

    Catalog entries carry a leading emoji ("⭐ SOFA-2 Scores"); it belongs to
    the picker UI, not to a table header or a chart legend.
    """

    text = str(value or "").strip()
    while text and not (text[0].isalnum() or "\u4e00" <= text[0] <= "\u9fff"):
        text = text[1:].lstrip()
    return text or str(value or "")


def _readable_id(module: str) -> str:
    """Fallback for a module the catalog does not know."""

    return module.replace("_", " ").title() if module else ""


def module_label_i18n(module: str) -> Dict[str, str]:
    """Both display names for one module, keyed ``en`` / ``zh``."""

    key = str(module or "")
    labels = concept_catalog.CONCEPT_GROUP_NAMES.get(key) or ()
    english = plain_label(labels[0]) if len(labels) > 0 else ""
    chinese = plain_label(labels[1]) if len(labels) > 1 else ""
    english = english or _readable_id(key)
    return {"en": english, "zh": chinese or english}


def module_label(module: str, lang: str = "en") -> str:
    labels = module_label_i18n(module)
    return labels.get(str(lang or "en")) or labels["en"]
