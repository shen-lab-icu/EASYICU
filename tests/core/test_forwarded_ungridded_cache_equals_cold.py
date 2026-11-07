"""A forwarded ungridded load never answers a gridded request in the same call.

A ``fun_itm`` built with ``fwd_concept('x')`` loads ``x`` with no grid
(``aggregate=None`` leaves the interval None) and the resolver stores that
table in the raw-concept cache. A later hourly request for ``x`` in the same
``load_concepts`` call must give exactly what a cold hourly load gives.

Before Codex delta4 D4-NATIVE-CACHE the reuse guard only compared two declared
grids, so a cached table with no interval was reused as if it were hourly.
eICU ``ett_gcs`` forwards ``rass``: in an extraction the neurological module
loaded gcs (via ett_gcs) before rass, and every hour with two or more RASS
charts held one raw chart instead of the hourly aggregate (G1 eICU sample:
507 cells in 185 stays). That path predates the time-axis sweep: plain
ICUTable results never carried a derived interval.

The resolver returns plain ``ICUTable`` and ``WinTbl`` (a ``TsTbl``) tables;
both forwarded shapes are loaded here in one call. A plain ``TsTbl`` is not a
resolver result, so its case seeds the raw cache the way the forwarding loader
writes it (legacy key, no grid). Synthetic rows only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.table import ICUTable, TsTbl, WinTbl
from easyicu.utils import compute_patient_ids_hash
from tests.support.aumc_source import DATA, aumc_db
from tests.support.eicu_source import DICTIONARY, _eicu_source

HOUR = pd.Timedelta(hours=1)
EICU_IDS = {"patientunitstayid": [1]}
AUMC_IDS = {"admissionid": [1]}
_PAIR = "Sedation Scale/Score/Goal"


def _frame(table) -> pd.DataFrame:
    return table.data.reset_index(drop=True)


def _load(resolver, names, source, ids):
    return resolver.load_concepts(
        names, source, merge=False, r_compatible=False, interval=HOUR,
        patient_ids=ids, concept_workers=1, verbose=False,
    )


def _eicu_two_rass_charts_in_one_hour(tmp_path):
    # Minute 5: RASS -4 (opens an ett_gcs window); minute 40: RASS -1. The
    # hourly value is the aggregate of both charts, not either raw chart.
    rows = []
    for offset, score in ((5, "-4"), (40, "-1"), (130, "-2")):
        rows += [
            (offset, "Scores", _PAIR, "Sedation Scale", "RASS"),
            (offset, "Scores", _PAIR, "Sedation Score", score),
        ]
    nurse = pd.DataFrame(rows, columns=[
        "nursingchartoffset", "nursingchartcelltypecat", "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname", "nursingchartvalue",
    ])
    nurse["nursingchartentryoffset"] = nurse["nursingchartoffset"]
    nurse["nursingchartid"] = np.arange(len(nurse), dtype=float)
    return _eicu_source(tmp_path, "eicu", nursecharting=nurse)


def test_icutable_forwarded_by_ett_gcs_equals_a_cold_hourly_load(tmp_path):
    source = _eicu_two_rass_charts_in_one_hour(tmp_path)
    dictionary = ConceptDictionary.from_json(DICTIONARY)

    cold = _load(ConceptResolver(dictionary), ["rass"], source, EICU_IDS)["rass"]
    assert type(cold) is ICUTable
    assert not cold.data["rass"].dropna().mod(1).eq(0).all(), "an hour must aggregate two charts"

    # ett_gcs first: its fwd_concept('rass') caches the ungridded rass table,
    # then the same call asks for rass on the hourly grid.
    warm = _load(ConceptResolver(dictionary), ["ett_gcs", "rass"], source, EICU_IDS)["rass"]

    pd.testing.assert_frame_equal(_frame(warm), _frame(cold), check_exact=True)
    assert (warm.index_column, warm.value_column) == (cold.index_column, cold.value_column)


def _aumc_dictionary_with_a_window_forward(tmp_path):
    # The HiRID ett_gcs pattern (fwd_concept on the mech_vent window table),
    # declared for AUMC so the synthetic AUMC source can carry it.
    payload = json.loads((DATA / "concept-dict.json").read_text())
    payload["vent_fwd"] = {
        "class_name": "lgl_cncpt", "target": "win_tbl", "aggregate": "max",
        "description": "test-only forward of mech_vent",
        "sources": {"aumc": [{
            "class_name": "fun_itm",
            "callback": "combine_callbacks(fwd_concept('mech_vent'), "
                        "transform_fun(comp_na(`==`, 'invasive')))",
        }]},
    }
    path = tmp_path / "dict_with_forward.json"
    path.write_text(json.dumps(payload))
    return ConceptDictionary.from_json(path)


def test_win_tbl_forwarded_load_equals_a_cold_hourly_load(tmp_path):
    db = aumc_db(tmp_path / "src", processes=[(30, 250, 9328, "Beademen")])
    dictionary = _aumc_dictionary_with_a_window_forward(tmp_path)

    native = ConceptResolver(dictionary).load_concepts(
        ["mech_vent"], db, merge=False, r_compatible=False, interval=None,
        patient_ids=AUMC_IDS, concept_workers=1, verbose=False,
    )["mech_vent"]
    assert isinstance(native, WinTbl) and isinstance(native, TsTbl) and native.interval is None

    cold = _load(ConceptResolver(dictionary), ["mech_vent"], db, AUMC_IDS)["mech_vent"]
    assert len(cold.data) > 1, "the window must expand over several hours"
    warm = _load(ConceptResolver(dictionary), ["vent_fwd", "mech_vent"], db, AUMC_IDS)["mech_vent"]

    pd.testing.assert_frame_equal(_frame(warm), _frame(cold), check_exact=True)


@pytest.mark.parametrize("legacy", [True, False])
def test_an_ungridded_plain_ts_tbl_cache_entry_is_not_reused_for_an_hourly_request(tmp_path, legacy):
    source = _eicu_two_rass_charts_in_one_hour(tmp_path)
    dictionary = ConceptDictionary.from_json(DICTIONARY)
    cold = _load(ConceptResolver(dictionary), ["rass"], source, EICU_IDS)["rass"]

    raw = ConceptResolver(dictionary).load_concepts(
        ["rass"], source, merge=False, r_compatible=False, interval=None, aggregate=None,
        patient_ids=EICU_IDS, concept_workers=1, verbose=False,
    )["rass"]
    entry = TsTbl(raw.data.copy(), id_vars=list(raw.id_columns), index_var=raw.index_column, interval=None)
    assert entry.interval is None

    resolver = ConceptResolver(dictionary)
    resolver._keep_cache_between_calls = True  # the seeded entry stands in for a forward
    resolver._store_raw_concept_cache(
        "rass", compute_patient_ids_hash(EICU_IDS), entry,
        aggregator=None, store_legacy=legacy,
    )
    warm = _load(resolver, ["rass"], source, EICU_IDS)["rass"]

    pd.testing.assert_frame_equal(_frame(warm), _frame(cold), check_exact=True)
