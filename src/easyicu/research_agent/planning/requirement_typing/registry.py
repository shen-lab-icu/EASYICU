"""Snapshot the coordinate registries used by the typed adapters."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from typing import TYPE_CHECKING

from easyicu.databases.time_origin import ICU_ADMISSION_ORIGIN, SIC_CARE_UNIT_ORIGIN
from easyicu.resources import (
    SIC_CASE_OVERLAY_CONCEPTS,
    SIC_CASE_OVERLAY_FILE,
    SIC_CASE_OVERLAY_SHA256,
    load_dictionary,
    package_path,
)

from ...canonical_json import canonical_sha256
from .contract import Binding, DictionarySnapshot

if TYPE_CHECKING:
    from ...schema import ConceptDescriptor


STUDY_POPULATION = Binding(role="population", concept_id="study_population")
_ANCHOR_ALIASES = {"icu_admit": ICU_ADMISSION_ORIGIN, "hospital_admit": "hospital_admission"}
_SOURCE_ANCHORS = frozenset({ICU_ADMISSION_ORIGIN, "hospital_admission", SIC_CARE_UNIT_ORIGIN})
_EVENT_ANCHORS = frozenset({"sep3", "vent_start", "vent_end", "rrt"})


@lru_cache(maxsize=4)
def registered_dictionary(source_name: str | None = None) -> DictionarySnapshot:
    dictionary = load_dictionary(include_sofa2=True, source_name=source_name)
    concepts = frozenset(dictionary.keys())
    anchors = _SOURCE_ANCHORS | (_EVENT_ANCHORS & concepts)
    files = {}
    for filename in ("concept-dict.json", "sofa2-dict.json"):
        with package_path(filename) as path:
            files[filename] = sha256(path.read_bytes()).hexdigest()
    if SIC_CASE_OVERLAY_CONCEPTS <= concepts:
        files[SIC_CASE_OVERLAY_FILE] = SIC_CASE_OVERLAY_SHA256
    # The shipped dictionary exposes extraction components and lineage.
    # Neither declares substitutability of one clinically scoped event for
    # another. A dictionary with such declarations can supply explicit edges.
    return DictionarySnapshot(
        version=canonical_sha256({
            "files": files, "concept_ids": sorted(concepts), "anchor_ids": sorted(anchors),
            "population_ids": [STUDY_POPULATION.concept_id],
        }),
        concept_ids=concepts,
        anchor_ids=anchors,
        population_ids={STUDY_POPULATION.concept_id},
    )


def registered_anchor(raw: str | None, dictionary: DictionarySnapshot) -> str | None:
    if raw is None:
        return None
    anchor = _ANCHOR_ALIASES.get(raw, raw)
    return anchor if anchor in dictionary.anchor_ids else None


def registered_concept_id(
    column: str, variable: ConceptDescriptor | None, dictionary: DictionarySnapshot,
) -> str | None:
    if variable and variable.concept_enrichment_degraded:
        return None
    if column in dictionary.concept_ids:
        return column
    # A source identity can bind a physical alias. Lineage with more than
    # that identity can describe a derived feature and cannot substitute.
    if variable and variable.source_concept in dictionary.concept_ids and (
        set(variable.derived_from_concepts) <= {variable.source_concept}
    ):
        return variable.source_concept
    return None
