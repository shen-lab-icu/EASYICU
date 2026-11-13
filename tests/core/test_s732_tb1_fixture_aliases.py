"""Shared synthetic channels satisfy every declared alias without duplicate rows."""

import numpy as np
import pandas as pd
import pytest

from easyicu.synthetic_source_tables import _events, definitions
from easyicu.synthetic_source_values import numeric_values, population


def _channel(database, table, first, second):
    config, _ = definitions(database)
    concepts = {"observed": {"sources": {database: [first]}},
                "restricted": {"sources": {database: [second]}}}
    return pd.concat(_events(database, config, concepts, population(12))[table], ignore_index=True)


def test_categorical_alias_supplies_valid_shared_labels():
    first = {"table": "listitems", "sub_var": "itemid", "ids": [990001],
             "callback": "transform_fun(set_val(TRUE))"}
    second = {**first, "callback": "apply_map(c(`supported` = 'known'), strict = TRUE)"}
    frame = _channel("aumc", "listitems", first, second)
    assert frame.value.eq("supported").all()
    assert not frame.duplicated(["admissionid", "measuredat", "itemid"]).any()


def test_route_aliases_share_an_eligible_infusion_channel():
    first = {"table": "pharma", "sub_var": "pharmaid", "ids": [990001],
             "require_regex": {"route": r"^(?:i.v.|iv-inf)$"}}
    second = {**first, "require_regex": {"route": r"^iv-inf$"}}
    frame = _channel("hirid", "pharma", first, second)
    assert frame.route.eq("iv-inf").all()
    assert not frame.duplicated(["patientid", "givenat", "pharmaid"]).any()


def test_later_fluid_alias_projects_boolean_and_category_gates():
    first = {"table": "drugitems", "sub_var": "itemid", "ids": [990001]}
    second = {**first, "val_var": "fluidin", "require_regex": {
        "isadditive": "^False$", "ordercategoryid": r"^(?:17)(?:\.0)?$"}}
    frame = _channel("aumc", "drugitems", first, second)
    assert frame.isadditive.eq(False).all()
    assert frame.ordercategoryid.eq(17).all()
    assert not frame.duplicated(["admissionid", "start", "itemid"]).any()


@pytest.mark.parametrize("parent,lower,upper", [("lact", 0., 50.), ("ph", 6., 8.5)])
def test_site_specific_values_use_the_parent_quantity_curve(parent, lower, upper):
    pop = population(12)
    index = np.arange(12)
    hours = np.full(12, 1.)
    definition = {"min": lower, "max": upper}
    canonical = numeric_values(parent, definition, pop, index, hours, np.random.default_rng(51))
    for suffix in ("_venous", "_mixed_venous"):
        site = numeric_values(parent + suffix, definition, pop, index, hours, np.random.default_rng(51))
        np.testing.assert_allclose(site, canonical)
