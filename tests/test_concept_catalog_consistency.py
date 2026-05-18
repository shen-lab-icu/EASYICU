from __future__ import annotations

from easyicu.resources import load_dictionary
from easyicu.webapp.concept_catalog import (
    COMPOSITE_CONCEPT_OUTPUT_SOURCES,
    CONCEPT_DICTIONARY,
    CONCEPT_GROUPS_INTERNAL,
    HIDDEN_DICTIONARY_CONCEPTS,
)


def test_web_catalog_groups_are_unique_and_complete() -> None:
    grouped = [concept for concepts in CONCEPT_GROUPS_INTERNAL.values() for concept in concepts]

    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(CONCEPT_DICTIONARY)


def test_web_catalog_aligns_with_merged_extraction_dictionary() -> None:
    dictionary = load_dictionary(include_sofa2=True)
    dict_concepts = set(dictionary.keys())
    web_concepts = set(CONCEPT_DICTIONARY)

    unresolved_web_concepts = web_concepts - dict_concepts - set(COMPOSITE_CONCEPT_OUTPUT_SOURCES)
    hidden_dict_concepts = dict_concepts - web_concepts

    assert unresolved_web_concepts == set()
    assert hidden_dict_concepts <= HIDDEN_DICTIONARY_CONCEPTS


def test_composite_output_sources_are_valid() -> None:
    dictionary = load_dictionary(include_sofa2=True)
    dict_concepts = set(dictionary.keys())
    special_sources = {"circ_failure_loader"}

    for output_concept, source_concept in COMPOSITE_CONCEPT_OUTPUT_SOURCES.items():
        assert output_concept in CONCEPT_DICTIONARY
        assert source_concept in dict_concepts or source_concept in special_sources
