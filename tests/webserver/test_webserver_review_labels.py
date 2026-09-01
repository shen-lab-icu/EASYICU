"""One set of display names for concept modules across the review surfaces.

Three copies of this lookup had drifted, and the drift reached the screen.
Cohort Statistics derived its label by title-casing the raw module id and
never consulted the concept catalog, so it rendered "Sofa2 Score" and
"Sepsis3 Sofa2" where Patient Review showed the catalog's "SOFA-2 Scores" and
"Sepsis-3 (SOFA-2 based)". Fourteen of nineteen modules were named differently
depending on the page, and Cohort Statistics copied its English string into the
``label_zh`` field, so the Chinese UI showed the mangled English too.

These are clinical proper nouns on a scientific surface; the catalog owns their
spelling.
"""

from __future__ import annotations

from pathlib import Path
import re

from easyicu.concept import catalog as concept_catalog
from easyicu.webserver import cohort_review, review_labels
from easyicu.webserver import patient_drilldown
from easyicu.webserver.patient_drilldown import coverage


WEBSERVER = Path(review_labels.__file__).resolve().parent


def test_every_review_surface_names_a_module_identically() -> None:
    mismatched = []
    for module in concept_catalog.CONCEPT_GROUP_NAMES:
        names = {
            "cohort_review": cohort_review._module_label(module),
            "patient_drilldown": patient_drilldown._module_label(module),
            "coverage": coverage._module_label(module, 0),
        }
        if len(set(names.values())) > 1:
            mismatched.append(f"{module}: {names}")
    assert mismatched == [], "\n".join(mismatched)


def test_labels_come_from_the_catalog_not_from_the_identifier() -> None:
    """Title-casing an id mangles the clinical names it contains."""

    assert review_labels.module_label("sofa2_score") == "SOFA-2 Scores"
    assert review_labels.module_label("sepsis3_sofa2") == "Sepsis-3 (SOFA-2 based)"
    assert review_labels.module_label("vitals") == "Vital Signs"
    # The surface that used to title-case must now agree.
    assert cohort_review._module_label("sofa2_score") == "SOFA-2 Scores"


def test_the_chinese_label_is_actually_chinese() -> None:
    zh = review_labels.module_label("sofa2_score", "zh")
    assert zh == "SOFA-2 评分"
    assert re.search(r"[一-鿿]", zh)
    assert cohort_review._module_label("sepsis3_sofa2", "zh") == "Sepsis-3 (基于SOFA-2)"

    # Every catalog module with a Chinese name must expose it, not the English.
    for module, names in concept_catalog.CONCEPT_GROUP_NAMES.items():
        if len(names) > 1 and re.search(r"[一-鿿]", names[1]):
            label = review_labels.module_label(module, "zh")
            assert re.search(r"[一-鿿]", label), f"{module} lost its zh name"


def test_the_catalog_emoji_stays_out_of_tables_and_legends() -> None:
    """The emoji belongs to the picker, not to a column header."""

    assert concept_catalog.CONCEPT_GROUP_NAMES["sofa2_score"][0].startswith("⭐")
    for module in concept_catalog.CONCEPT_GROUP_NAMES:
        for lang in ("en", "zh"):
            label = review_labels.module_label(module, lang)
            assert label[:1].isalnum() or re.match(r"[一-鿿]", label[:1])


def test_an_unknown_module_still_reads_as_a_name() -> None:
    assert review_labels.module_label("totally_unknown") == "Totally Unknown"
    assert review_labels.module_label("totally_unknown", "zh") == "Totally Unknown"
    assert review_labels.module_label("") == ""
    assert review_labels.module_label_i18n(None) == {"en": "", "zh": ""}


def test_an_unknown_language_falls_back_to_english() -> None:
    assert review_labels.module_label("vitals", "fr") == "Vital Signs"
    assert review_labels.module_label("vitals", "") == "Vital Signs"


def test_the_label_derivation_lives_in_exactly_one_module() -> None:
    """A second copy is how the surfaces drifted apart in the first place."""

    offenders = []
    for path in sorted(WEBSERVER.rglob("*.py")):
        if path.name == "review_labels.py":
            continue
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(WEBSERVER)
        # The emoji-stripping loop, copied verbatim into two modules.
        if "\\u4e00" in source and "isalnum()" in source:
            offenders.append(f"{rel} re-implements the label cleanup")
        # Title-casing a module id instead of asking the catalog.
        if re.search(r'module\.replace\("_", " "\)\.title\(\)', source):
            offenders.append(f"{rel} derives a label from the identifier")
    assert offenders == [], "\n".join(offenders)
