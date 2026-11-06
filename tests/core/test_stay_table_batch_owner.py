"""The stay-origin table has one owner per patient batch (lead-r13 #13 item 4).

A streamed batch read eICU ``patient`` up to seven times (whole table, every
column, for the episode window; filtered, for per-kg rate weights) and its
dependency loads repeated it on fresh resolvers. The owner reads it once per
batch, restricted to the batch's stays, with every column, and serves row and
column subsets that equal the filtered reads they replace.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.stay_tables import clear_stay_tables, stay_table_batch, stay_table_reads
from easyicu.config import load_src_cfg
from easyicu.datasource import FilterOp, FilterSpec, ICUDataSource
from tests.support.bucket_storage import write_fixture_parquet

DICTIONARY = Path(__file__).resolve().parents[2] / "src/easyicu/data/concept-dict.json"
STAYS = [1, 2, 3, 4]


def _source(root: Path) -> ICUDataSource:
    root.mkdir(parents=True, exist_ok=True)
    write_fixture_parquet(pd.DataFrame({
        "patientunitstayid": STAYS,
        "patienthealthsystemstayid": [11, 12, 13, 14],
        "uniquepid": ["a", "b", "c", "d"],
        "age": ["60", "70", "> 89", "45"], "gender": ["Male", "Female", "Male", "Female"],
        "hospitalid": [1, 1, 2, 2], "unitvisitnumber": [1, 1, 1, 1],
        "hospitaladmitoffset": [0, -60, 0, -30],
        "hospitaldischargeoffset": [72 * 60] * 4,
        "unitdischargeoffset": [48 * 60, 24 * 60, 36 * 60, 12 * 60],
        "unitdischargestatus": ["Alive"] * 4, "hospitaldischargestatus": ["Alive"] * 4,
        "admissionweight": [80.0, 65.5, None, 90.0],
    }), root / "patient.parquet", index=False)
    offsets = [30, 90, 50 * 60, 13 * 60, 37 * 60]
    (root / "vitalperiodic").mkdir(exist_ok=True)
    write_fixture_parquet(pd.DataFrame({
        "vitalperiodicid": range(len(offsets) * len(STAYS)),
        "patientunitstayid": [s for s in STAYS for _ in offsets],
        "observationoffset": offsets * len(STAYS),
        "heartrate": [70.0, 75.0, 80.0, 85.0, 90.0] * len(STAYS),
    }), root / "vitalperiodic" / "part.parquet", index=False)
    return ICUDataSource(load_src_cfg("eicu"), base_path=root, enable_cache=False)


def _hr(source, stays):
    resolver = ConceptResolver(ConceptDictionary.from_json(DICTIONARY))
    return resolver, resolver.load_concepts(
        ["hr"], source, merge=True, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"patientunitstayid": stays},
        concept_workers=1, verbose=False,
    )


@pytest.fixture(autouse=True)
def _fresh_owner():
    clear_stay_tables()
    yield
    clear_stay_tables()


def test_one_read_per_batch_across_resolvers(tmp_path):
    source = _source(tmp_path / "eicu")
    before = stay_table_reads()
    with stay_table_batch():  # a streamed export's batch scope
        _hr(source, [1, 2])
        # A dependency load of the same batch on a fresh resolver and source.
        _hr(ICUDataSource(load_src_cfg("eicu"), base_path=tmp_path / "eicu", enable_cache=False), [1, 2])
        assert stay_table_reads() - before == 1
        _hr(source, [3, 4])  # the next batch
        assert stay_table_reads() - before == 2


def test_batch_clip_equals_a_clip_from_the_whole_table(tmp_path):
    source = _source(tmp_path / "eicu")
    _, batched = _hr(source, [2, 4])
    clear_stay_tables()
    whole = ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        ["hr"], source, merge=True, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids=None,
        concept_workers=1, verbose=False,
    )
    expected = whole.loc[whole["patientunitstayid"].isin([2, 4])].reset_index(drop=True)
    pd.testing.assert_frame_equal(batched.reset_index(drop=True), expected, check_exact=True)
    # Stays 2 and 4 were discharged at 24 h and 12 h: later rows are clipped.
    assert set(batched["patientunitstayid"]) == {2, 4}


def test_subset_equals_the_filtered_read_it_replaces(tmp_path):
    source = _source(tmp_path / "eicu")
    resolver, _ = _hr(source, STAYS)
    owned = resolver._stay_table(
        source, "patient", stays=[4, 2], columns=["patientunitstayid", "admissionweight"]
    )
    # ``load_table`` may return more columns than requested; the owner
    # returns exactly the requested ones, with the same rows and values.
    direct = source.load_table(
        "patient", columns=["patientunitstayid", "admissionweight"],
        filters=[FilterSpec("patientunitstayid", FilterOp.IN, [4, 2])], verbose=False,
    ).data[["patientunitstayid", "admissionweight"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(owned, direct, check_exact=True)


def test_a_narrow_first_reader_cannot_narrow_a_later_one(tmp_path):
    source = _source(tmp_path / "eicu")
    resolver, _ = _hr(source, STAYS)
    resolver._stay_table(source, "patient", columns=["patientunitstayid"])
    later = resolver._stay_table(source, "patient", columns=["patientunitstayid", "unitdischargeoffset"])
    assert list(later.columns) == ["patientunitstayid", "unitdischargeoffset"]
    assert later["unitdischargeoffset"].notna().all()


def test_stays_outside_the_batch_are_read_directly(tmp_path):
    source = _source(tmp_path / "eicu")
    resolver, _ = _hr(source, [1, 2])
    before = stay_table_reads()
    frame = resolver._stay_table(source, "patient", stays=[3], columns=["patientunitstayid"])
    assert frame["patientunitstayid"].tolist() == [3]
    assert stay_table_reads() == before  # not cached under the batch scope
