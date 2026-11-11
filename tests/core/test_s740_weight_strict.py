"""S740: leakage-free stay weight (weight_admit_strict) beside the unchanged weight concept."""

import numpy as np
import pandas as pd
import pytest

from easyicu.api import EXTRACT_MODULE_ORDER, EXTRACT_MODULES
from easyicu.concept.catalog import CONCEPT_GROUPS_INTERNAL, HIDDEN_DICTIONARY_CONCEPTS, WEIGHT_STRICT_CONCEPTS
from easyicu.concept.weight_strict import FIELDS, SOURCES, select_strict_weight
from easyicu.resources import load_dictionary


def _obs(rows):
    return pd.DataFrame(rows, columns=["sid", "channel", "rank", "time_h", "kg"])


def _pick(rows, stays):
    out = select_strict_weight(_obs(rows), pd.Series(stays)).set_index("sid")
    return {sid: (None if pd.isna(r.weight_admit_strict) else r.weight_admit_strict,
                  r.weight_admit_strict_source,
                  None if pd.isna(r.weight_admit_strict_time_h) else r.weight_admit_strict_time_h)
            for sid, r in out.iterrows()}


def test_rule_order_and_never_a_median():
    rows = [
        (1, "admission", 0, 5.0, 80.0), (1, "admission", 0, 2.0, 82.0), (1, "fallback", 0, -3.0, 70.0),
        (2, "admission_lbs", 0, 1.0, 90.0), (2, "admission_lbs", 0, 0.5, 91.0),
        (3, "fallback", 0, -10.0, 60.0), (3, "fallback", 0, -2.0, 61.0), (3, "fallback", 0, 4.0, 99.0),
        (4, "fallback", 0, 30.0, 75.0), (4, "fallback", 0, 6.0, 76.0),
    ]
    picked = _pick(rows, [1, 2, 3, 4, 5])
    assert picked[1] == (82.0, "admission", 2.0)        # earliest admission reading, not the median
    assert picked[2] == (91.0, "admission_lbs", 0.5)
    assert picked[3] == (61.0, "pre_icu_last", -2.0)    # last reading before ICU admission
    assert picked[4] == (76.0, "charted_first", 6.0)    # first reading after ICU admission
    assert picked[5] == (None, "none", None)            # cohort stay without a usable reading


def test_admission_beats_admission_lbs_and_out_of_range_readings_do_not_count():
    rows = [(1, "admission_lbs", 0, 0.0, 70.0), (1, "admission", 0, 9.0, 72.0),
            (2, "admission", 0, 0.0, 0.5), (2, "fallback", 0, 1.0, 66.0),
            (3, "admission", 0, 0.0, 600.0), (3, "fallback", 0, np.nan, 66.0)]
    picked = _pick(rows, [1, 2, 3])
    assert picked[1] == (72.0, "admission", 9.0)
    assert picked[2] == (66.0, "charted_first", 1.0)
    assert picked[3] == (None, "none", None)


def test_ties_keep_the_better_fallback_channel_then_the_smaller_value():
    rows = [(1, "fallback", 1, 2.0, 80.0), (1, "fallback", 0, 2.0, 85.0),
            (2, "fallback", 0, -1.0, 71.0), (2, "fallback", 0, -1.0, 70.0)]
    picked = _pick(rows, [1, 2])
    assert picked[1] == (85.0, "charted_first", 2.0)
    assert picked[2] == (70.0, "pre_icu_last", -1.0)


def test_weight_strict_is_an_opt_in_module_and_weight_is_unchanged():
    assert tuple(EXTRACT_MODULES["weight_strict"]) == WEIGHT_STRICT_CONCEPTS == FIELDS
    assert "weight_strict" not in EXTRACT_MODULE_ORDER
    grouped = {c for members in CONCEPT_GROUPS_INTERNAL.values() for c in members}
    assert set(WEIGHT_STRICT_CONCEPTS) <= HIDDEN_DICTIONARY_CONCEPTS
    assert not set(WEIGHT_STRICT_CONCEPTS) & grouped
    dictionary = load_dictionary()
    assert list(dictionary["weight_admit_strict_source"].levels) == list(SOURCES)
    for concept in WEIGHT_STRICT_CONCEPTS:
        assert dictionary[concept].target == "id_tbl"
        for database in ("miiv", "aumc", "hirid", "sic", "eicu"):
            (source,) = dictionary[concept].sources[database]
            assert source.callback == "weight_admit_strict_fields"
    # The published weight keeps its channel-median definition.
    assert dictionary["weight"].sources["miiv"][0].ids == [226512]


@pytest.mark.parametrize("bad", [None, {"stay_id": [1], "hadm_id": [2]}])
def test_patient_scope_must_name_the_stay_identifier(bad):
    from easyicu.concept.weight_strict import _selected_ids

    if bad is None:
        assert _selected_ids(None, "stay_id") is None
    else:
        assert _selected_ids(bad, "stay_id") == [1]
    with pytest.raises(ValueError):
        _selected_ids({"hadm_id": [2]}, "stay_id")


class _AumcSource:
    """load_table as the AUMC data source returns it: times already in minutes (ms / 60000)."""

    class config:
        name = "aumc"

    def __init__(self, tables):
        self.tables = tables

    def load_table(self, table, columns=None, filters=None, verbose=False):
        return self.tables[table].loc[:, list(columns)].copy()


def test_aumc_times_from_load_table_are_minutes():
    from easyicu.concept import weight_strict

    source = _AumcSource({
        "admissions": pd.DataFrame({"admissionid": [1, 2], "admittedat": [600.0, 0.0], "weightgroup": [None, None]}),
        "numericitems": pd.DataFrame({"admissionid": [1, 2], "measuredat": [480.0, 90.0], "itemid": [6638, 6638],
                                      "value": [70.0, 80.0]}),
    })
    weight_strict._CACHE.clear()
    out, id_column = weight_strict.strict_weight_fields(source, None)
    out = out.set_index(id_column)
    assert (out.loc[1, "weight_admit_strict_source"], out.loc[1, "weight_admit_strict_time_h"]) == ("pre_icu_last", -2.0)
    assert (out.loc[2, "weight_admit_strict_source"], out.loc[2, "weight_admit_strict_time_h"]) == ("charted_first", 1.5)
    weight_strict._CACHE.clear()
