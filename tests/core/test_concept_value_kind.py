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
    default_aggregate,
)
from tests.support.blood_gas_sources import DICTIONARY


def _definition(class_name=None, minimum=None, maximum=None, units=None, name="x"):
    return SimpleNamespace(name=name, class_name=class_name, minimum=minimum, maximum=maximum, units=units)


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


@pytest.mark.parametrize(
    ("units", "name", "kind"),
    [(["boolean"], "x", "logical"), (["mL"], "x", "numeric"), (None, "sep3", "logical"),
     (None, "sofa", "numeric"), (None, "avpu", "text")],
)
def test_a_recipe_takes_its_unit_or_documented_output_type(units, name, kind):
    definition = _definition("rec_cncpt", units=units, name=name)
    assert declared_value_kind(definition, database="eicu") == kind


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
    assert undecided == []


@pytest.mark.parametrize("path", [DICTIONARY, DICTIONARY.with_name("sofa2-dict.json")])
def test_every_shipped_concept_declares_its_kind(path):
    dictionary = ConceptDictionary.from_json(path)
    undecided = []
    for name in dictionary.keys():
        try:
            declared_value_kind(dictionary[name], database="any")
        except ConceptValueKindUndeclared:
            undecided.append(name)
    assert undecided == []


def test_text_concepts_stay_text_whatever_their_values_look_like():
    values = pd.Series(["1", "2", "mild", None], dtype=object)
    pd.testing.assert_series_equal(coerce_declared_values(values, "text"), values)


def test_numeric_parsing_matches_the_sampled_route_parser():
    # The sampled route parsed with pd.to_numeric(errors="coerce"); every form
    # it accepted or refused comes out the same.
    forms = pd.Series(
        ["1.5", " 2 ", "+3", "-4", "1e3", ".5", "7.", "0x10", "1,5", "5 mg", "", " ", "nan",
         "inf", "-inf", "NA", "Geïntubeerd", 6, 2.25, True, False, None],
        dtype=object,
    )
    old_route = pd.to_numeric(forms, errors="coerce")
    pd.testing.assert_series_equal(coerce_declared_values(forms, "numeric"), old_route)


def test_logical_values_keep_booleans_read_numbers_and_drop_text():
    values = pd.Series([True, None, 1.0, 0, "0", "x", 2, False], dtype=object)
    assert coerce_declared_values(values, "logical").tolist() == [
        True, pd.NA, True, False, False, pd.NA, True, False,
    ]
    # Wherever the sampled route parsed a logical column, the truth value agrees.
    old_route = pd.to_numeric(values, errors="coerce")
    parsed = old_route.notna()
    assert coerce_declared_values(values, "logical")[parsed].tolist() == old_route[parsed].ne(0).tolist()


@pytest.mark.parametrize(
    ("kind", "nullable", "aggregate"),
    [("numeric", False, "median"), ("logical", False, "any"), ("logical", True, "max"), ("text", False, "first")],
)
def test_default_aggregate_follows_the_kind(kind, nullable, aggregate):
    assert default_aggregate(kind, nullable_status=nullable) == aggregate


def test_a_classless_unbounded_concept_refuses_text_and_asks_for_a_class():
    from easyicu.concept.value_kind import implicitly_numeric

    definition = _definition(None, name="custom_score")
    assert implicitly_numeric(definition)
    values = pd.Series(["3", "mild", None], dtype=object)
    with pytest.raises(ConceptValueKindUndeclared, match="'custom_score' on 'miiv'.*declare class"):
        coerce_declared_values(values, "numeric", refuse_text_for=(definition, "miiv"))
    # Numeric text parses without complaint.
    parsed = coerce_declared_values(pd.Series(["3", " 4 ", None], dtype=object), "numeric",
                                    refuse_text_for=(definition, "miiv"))
    assert parsed.tolist()[:2] == [3, 4]


@pytest.mark.parametrize("declaration", [{"class_name": "num_cncpt"}, {"minimum": 1.0}])
def test_a_declared_numeric_concept_turns_text_into_missing(declaration):
    from easyicu.concept.value_kind import implicitly_numeric

    assert not implicitly_numeric(_definition(**declaration))
    values = pd.Series(["3", "Geïntubeerd", None], dtype=object)
    assert coerce_declared_values(values, "numeric").isna().tolist() == [False, True, True]


@pytest.mark.parametrize("include_sofa2", [False, True])
def test_no_shipped_concept_is_numeric_only_by_default(include_sofa2):
    from easyicu.concept.value_kind import implicitly_numeric
    from easyicu.resources import load_dictionary

    dictionary = load_dictionary(include_sofa2=include_sofa2)
    # The former unbounded raw measurements (bnd, bun_raw, crea_raw, etco2_raw,
    # hct_raw, hgb_raw, inr_pt, minute_vol_raw, myoglobin_raw, na_raw, ph_raw,
    # temp_raw) declare num_cncpt (B-IMPLICIT-GATE): the publish-time refusal
    # of a classless concept can only fire on a user's custom concept.
    assert sorted(name for name in dictionary.keys() if implicitly_numeric(dictionary[name])) == []
    for name in ("bnd", "inr_pt", "temp_raw"):
        assert dictionary[name].class_name == "num_cncpt"
    for name in dictionary.keys():
        declared_value_kind(dictionary[name], database="any")


def test_export_keeps_a_text_concept_text_when_its_batch_holds_only_codes():
    from easyicu.api.extraction import _normalise_module_frame_for_parquet

    frame = pd.DataFrame({
        "stay_id": [1, 2], "charttime": [0.0, 1.0],
        "icu_unit_type": pd.Series(["2", "5"], dtype=object),
        "tgcs": pd.Series(["14", "Unable to score due to medication"], dtype=object),
    })
    out = _normalise_module_frame_for_parquet(frame, ["icu_unit_type", "tgcs"], reorder=False)
    assert out["icu_unit_type"].tolist() == ["2", "5"]
    assert out["tgcs"].isna().tolist() == [False, True]
