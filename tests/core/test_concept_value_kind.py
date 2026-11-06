"""A concept's value kind comes from its declaration, never from its batch.

The resolver used to parse a non-numeric value column only when most of the
batch's first 100 values parsed. ``declared_value_kind`` replaces that with
the dictionary's class and numeric bounds; an undecidable declaration is a
typed refusal. Every value here is synthetic.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary
from easyicu.concept.value_kind import (
    ConceptValueKindUndeclared,
    coerce_declared_values,
    declared_value_kind,
)
from tests.support.blood_gas_sources import DICTIONARY


def _definition(class_name=None, minimum=None, maximum=None):
    return SimpleNamespace(name="x", class_name=class_name, minimum=minimum, maximum=maximum)


@pytest.mark.parametrize(
    ("class_name", "minimum", "kind"),
    [
        (None, None, "numeric"),
        ("num_cncpt", None, "numeric"),
        (["unt_cncpt", "num_cncpt"], 1.0, "numeric"),
        ("rec_cncpt", 0.0, "numeric"),
        ("lgl_cncpt", None, "logical"),
        ("fct_cncpt", None, "text"),
        ("chr_cncpt", None, "text"),
    ],
)
def test_kind_follows_the_declared_class_and_bounds(class_name, minimum, kind):
    assert declared_value_kind(_definition(class_name, minimum), database="aumc") == kind


@pytest.mark.parametrize("class_name", ["rec_cncpt", ["lgl_cncpt", "fct_cncpt"], "unknown_cncpt"])
def test_an_undecidable_declaration_is_refused_by_name(class_name):
    with pytest.raises(ConceptValueKindUndeclared, match="'x' on 'aumc'"):
        declared_value_kind(_definition(class_name), database="aumc")


@pytest.mark.parametrize("path", [DICTIONARY, DICTIONARY.with_name("sofa2-dict.json")])
def test_every_shipped_source_concept_declares_its_kind(path):
    dictionary = ConceptDictionary.from_json(path)
    undecided = []
    for name in dictionary.keys():
        definition = dictionary[name]
        if not definition.sources:
            continue
        try:
            declared_value_kind(definition, database="any")
        except ConceptValueKindUndeclared:
            undecided.append(name)
    # Unbounded recipes whose sources are callbacks returning typed numbers or
    # booleans; they would be refused, not guessed, if text ever reached them.
    assert sorted(undecided) == (["fluid_balance", "supp_o2"] if path == DICTIONARY else [])


def test_text_concepts_stay_text_whatever_their_values_look_like():
    values = pd.Series(["1", "2", "mild", None], dtype=object)
    pd.testing.assert_series_equal(coerce_declared_values(values, "text"), values)


def test_numeric_parsing_matches_the_sampled_route_parser():
    # The sampled route parsed with pd.to_numeric(errors="coerce"); every form
    # it accepted or refused must come out the same.
    forms = pd.Series(
        ["1.5", " 2 ", "+3", "-4", "1e3", ".5", "7.", "0x10", "1,5", "5 mg", "", " ", "nan",
         "inf", "-inf", "NA", "Geïntubeerd", 6, 2.25, True, False, None],
        dtype=object,
    )
    old_route = pd.to_numeric(forms, errors="coerce")
    for kind in ("numeric", "logical"):
        pd.testing.assert_series_equal(coerce_declared_values(forms, kind), old_route)
