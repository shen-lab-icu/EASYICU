"""Snapshot the coordinate registries used by the typed adapters."""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from typing import TYPE_CHECKING

from easyicu.clinical_contracts import load_clinical_contracts
from easyicu.databases.time_origin import ICU_ADMISSION_ORIGIN, SIC_CARE_UNIT_ORIGIN
from easyicu.resources import (
    SIC_CASE_OVERLAY_CONCEPTS,
    SIC_CASE_OVERLAY_FILE,
    SIC_CASE_OVERLAY_SHA256,
    load_dictionary,
    package_path,
)

from ...canonical_json import canonical_sha256
from ...contracts.prediction_validation import EXTERNAL_ACTION
from ...contracts.product_files import RUN_DELIVERABLES, RUN_DELIVERY_SCHEMA_VERSION
from .contract import Binding, DictionarySnapshot, Grading, Method, RequirementKey, Unit

if TYPE_CHECKING:
    from easyicu.concept.schema import ConceptDictionary

    from ...schema import ConceptDescriptor


STUDY_POPULATION = Binding(role="population", concept_id="study_population")
_ANCHOR_ALIASES = {"icu_admit": ICU_ADMISSION_ORIGIN, "hospital_admit": "hospital_admission"}
_SOURCE_ANCHORS = frozenset({ICU_ADMISSION_ORIGIN, "hospital_admission", SIC_CARE_UNIT_ORIGIN})
_EVENT_ANCHORS = frozenset({"sep3", "vent_start", "vent_end", "rrt"})
_METHOD_POLICIES = frozenset({
    "binary_effect_measure", "descriptive.exposure_outcome_distribution", "model_term",
    "multiple_imputation", EXTERNAL_ACTION,
})
_GRADING_SOURCE_LABELS = {
    ("kdigo_aki_2012", "2012", "KDIGO 2012 AKI guideline", "kdigo_creat"): "kdigo 2012",
}


@lru_cache(maxsize=4)
def _dictionary(source_name: str | None) -> ConceptDictionary:
    return load_dictionary(include_sofa2=True, source_name=source_name)


@lru_cache(maxsize=1)
def contract_registry_sha256() -> str:
    return canonical_sha256({
        "intake": "easyicu.requirement_intake/2",
        "grading": sorted((item.contract_id, item.version, item.source_id, item.concepts)
                          for item in load_clinical_contracts().values()),
        "grading_source_labels": sorted((*key, label) for key, label in _GRADING_SOURCE_LABELS.items()),
        "method_policies": sorted(_METHOD_POLICIES),
        "delivery": (RUN_DELIVERY_SCHEMA_VERSION, sorted(RUN_DELIVERABLES)),
    })


def unregistered_fields(
    key: RequirementKey, dictionary: DictionarySnapshot, *, units: frozenset[str],
) -> tuple[str, ...]:
    fields = []
    for index, binding in enumerate(key.bindings):
        ids = {
            "time_anchor": dictionary.anchor_ids,
            "population": dictionary.concept_ids if key.facet.kind == "cohort_predicate" else dictionary.population_ids,
        }.get(binding.role, dictionary.concept_ids)
        if binding.concept_id not in ids:
            fields.append(f"bindings[{index}].concept_id")
    facet = key.facet
    if isinstance(facet, Method) and facet.action_id not in _METHOD_POLICIES:
        fields.append("facet.action_id")
    if isinstance(facet, Unit) and facet.code not in units:
        fields.append("facet.code")
    if isinstance(facet, Grading):
        owner = load_clinical_contracts().get(facet.contract_id)
        if owner is None or any(
            binding.concept_id not in owner.concepts for binding in key.bindings if binding.role != "time_anchor"
        ):
            fields.append("facet.contract_id")
        elif owner.version != facet.version:
            fields.append("facet.version")
    return tuple(fields)


@lru_cache(maxsize=4)
def registered_unit_codes(source_name: str | None) -> frozenset[str]:
    dictionary = _dictionary(source_name)
    return frozenset(unit for definition in dictionary.values() for unit in (definition.units or ()))


@lru_cache(maxsize=4)
def registered_dictionary(source_name: str | None = None) -> DictionarySnapshot:
    dictionary = _dictionary(source_name)
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
            "unit_codes": sorted(registered_unit_codes(source_name)),
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
    if variable is None:
        return column if column in dictionary.concept_ids else None
    if variable.concept_enrichment_degraded:
        return None
    concept = column if column in dictionary.concept_ids else variable.source_concept
    if (
        concept in dictionary.concept_ids
        and variable.source_concept in (None, concept)
        and set(variable.derived_from_concepts) <= {concept}
    ):
        return concept
    return None


def registered_grading(variable: ConceptDescriptor, dictionary: DictionarySnapshot) -> Grading | None:
    definition = variable.clinical_definition
    if definition is None:
        return None
    owner = load_clinical_contracts().get(definition.contract_id)
    if (
        owner and registered_concept_id(variable.name, variable, dictionary) in owner.concepts
        and (definition.version, definition.source_id) == (owner.version, owner.source_id)
    ):
        return Grading(contract_id=owner.contract_id, version=owner.version)
    return None


def registered_grading_source_label(facet: Grading, concept_id: str) -> str | None:
    owner = load_clinical_contracts().get(facet.contract_id)
    if owner and facet.version == owner.version and concept_id in owner.concepts:
        return _GRADING_SOURCE_LABELS.get((owner.contract_id, owner.version, owner.source_id, concept_id))
    return None
