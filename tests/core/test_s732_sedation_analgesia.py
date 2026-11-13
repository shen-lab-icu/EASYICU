"""S732 B4: sedation/analgesia exposures in the opt-in sedation_analgesia module."""

import pytest

from easyicu.api import EXTRACT_MODULE_ORDER, EXTRACT_MODULES
from easyicu.concept.catalog import (
    CONCEPT_DICTIONARY,
    CONCEPT_GROUPS_INTERNAL,
    HIDDEN_DICTIONARY_CONCEPTS,
    SEDATION_ANALGESIA_CONCEPTS,
)
from easyicu.resources import load_dictionary

CENSUS = {
    "remifentanil": {"aumc", "hirid", "sic"},
    "sufentanil": {"aumc", "hirid", "sic"},
    "piritramide": {"aumc", "sic"},
    "esketamine": {"aumc", "sic"},
    "hydromorphone": {"miiv", "sic"},
    "methadone": {"miiv", "aumc", "hirid", "sic"},
    "oxycodone": {"aumc", "hirid", "sic"},
    "tramadol": {"aumc", "hirid", "sic"},
}


def test_module_is_opt_in_and_its_concepts_are_hidden():
    assert EXTRACT_MODULES["sedation_analgesia"] == list(SEDATION_ANALGESIA_CONCEPTS)
    assert "sedation_analgesia" not in EXTRACT_MODULE_ORDER
    assert set(SEDATION_ANALGESIA_CONCEPTS) == set(CENSUS)
    grouped = {c for members in CONCEPT_GROUPS_INTERNAL.values() for c in members}
    for concept in SEDATION_ANALGESIA_CONCEPTS:
        assert concept in HIDDEN_DICTIONARY_CONCEPTS
        assert concept not in grouped
        assert concept not in CONCEPT_DICTIONARY


@pytest.mark.parametrize("concept", sorted(CENSUS))
def test_exposures_are_boolean_hourly_max_in_their_census_databases(concept):
    definition = load_dictionary()[concept]
    assert definition.class_name == "lgl_cncpt" and definition.aggregate == "max"
    assert set(definition.sources) == CENSUS[concept]
    for database, sources in definition.sources.items():
        for source in sources:
            if source.table in {"drugitems", "inputevents"}:
                assert source.dur_var in {"stop", "endtime"}, (database, source.table)
            if source.table == "medication":
                assert source.callback == "sicdb_active_exposure"


def test_neuraxial_sufentanil_and_ketamine_racemate_are_not_mapped():
    raw = load_dictionary()
    assert 21242 not in set(raw["sufentanil"].sources["aumc"][0].ids)
    assert set(raw["esketamine"].sources["aumc"][0].ids) == {9018}
    assert set(raw["esketamine"].sources["sic"][0].ids) == {1578}



def test_unreviewed_emar_sources_are_excluded():
    for concept in CENSUS:
        for sources in load_dictionary()[concept].sources.values():
            assert all(source.table != "emar" for source in sources)
