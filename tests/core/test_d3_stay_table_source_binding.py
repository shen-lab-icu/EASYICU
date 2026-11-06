"""The batch stay-table owner is keyed by the files it read, not by a path (Codex delta3 D3-STAY-CACHE).

The owner used to key a held frame by database + installation path only,
and a warm hit returned it without asking the storage owner again: a new
source object over the same path, whose stay table had been replaced or
removed, got the old discharge bounds and weights. Now every lookup
resolves the table's validated files and stats them:
- another file set misses;
- the same files with another size or mtime raise
  ``stay_table_source_changed``;
- a validated file that is gone raises ``stay_table_source_missing``.
One read per table per batch still holds. Rows are synthetic.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.errors import ConceptTableReadError
from easyicu.concept.stay_tables import (
    SourceBinding,
    StayTableSourceChanged,
    clear_stay_tables,
    source_binding,
    stay_table_frame,
    stay_table_reads,
)
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.storage_layout import StorageIntegrityError
from tests.support.bucket_storage import write_fixture_parquet

DICTIONARY = Path(__file__).resolve().parents[2] / "src/easyicu/data/concept-dict.json"
STAYS = [1, 2, 3, 4]


def _patient(discharge_hours) -> pd.DataFrame:
    return pd.DataFrame({
        "patientunitstayid": STAYS,
        "patienthealthsystemstayid": [11, 12, 13, 14],
        "uniquepid": ["a", "b", "c", "d"],
        "age": ["60", "70", "> 89", "45"], "gender": ["Male", "Female", "Male", "Female"],
        "hospitalid": [1, 1, 2, 2], "unitvisitnumber": [1, 1, 1, 1],
        "hospitaladmitoffset": [0, -60, 0, -30],
        "hospitaldischargeoffset": [72 * 60] * 4,
        "unitdischargeoffset": [h * 60 for h in discharge_hours],
        "unitdischargestatus": ["Alive"] * 4, "hospitaldischargestatus": ["Alive"] * 4,
        "admissionweight": [80.0, 65.5, None, 90.0],
    })


def _install(root: Path, discharge_hours=(48, 24, 36, 12)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_fixture_parquet(_patient(discharge_hours), root / "patient.parquet", index=False)
    offsets = [30, 90, 13 * 60, 20 * 60, 37 * 60]
    (root / "vitalperiodic").mkdir(exist_ok=True)
    write_fixture_parquet(pd.DataFrame({
        "vitalperiodicid": range(len(offsets) * len(STAYS)),
        "patientunitstayid": [s for s in STAYS for _ in offsets],
        "observationoffset": offsets * len(STAYS),
        "heartrate": [70.0, 75.0, 80.0, 85.0, 90.0] * len(STAYS),
    }), root / "vitalperiodic" / "part.parquet", index=False)
    return root


def _source(root: Path) -> ICUDataSource:
    return ICUDataSource(load_src_cfg("eicu"), base_path=root, enable_cache=False)


def _hr(source, stays):
    return ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        ["hr"], source, merge=True, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"patientunitstayid": stays},
        concept_workers=1, verbose=False,
    )


@pytest.fixture(autouse=True)
def _fresh_owner():
    clear_stay_tables()
    yield
    clear_stay_tables()


def test_the_same_files_are_read_once_per_batch_across_sources(tmp_path):
    root = _install(tmp_path / "eicu")
    before = stay_table_reads()
    _hr(_source(root), [1, 2])
    _hr(_source(root), [1, 2])
    assert stay_table_reads() - before == 1


def test_a_replaced_stay_table_within_the_batch_is_refused_not_reused(tmp_path):
    root = _install(tmp_path / "eicu")
    first = _hr(_source(root), [2, 4])
    # Same path, new bytes: stay 2 now stays 40 h instead of 24 h.
    write_fixture_parquet(_patient((48, 40, 36, 12)), root / "patient.parquet", index=False)
    with pytest.raises(StorageIntegrityError) as error:
        _hr(_source(root), [2, 4])
    assert error.value.reason == "stay_table_source_changed"
    assert len(first)  # the first batch read was ordinary


def test_a_touched_stay_table_within_the_batch_is_refused(tmp_path):
    root = _install(tmp_path / "eicu")
    _hr(_source(root), [1, 2])
    stat = (root / "patient.parquet").stat()
    os.utime(root / "patient.parquet", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(StayTableSourceChanged) as error:
        _hr(_source(root), [1, 2])
    assert error.value.reason == "stay_table_source_changed"


def test_a_removed_stay_table_is_never_served_from_the_held_frame(tmp_path):
    root = _install(tmp_path / "eicu")
    _hr(_source(root), [1, 2])
    (root / "patient.parquet").unlink()
    # An unmanifested installation no longer lists the table, so the binding
    # is new and the read fails typed; the held frame is never served.
    with pytest.raises(ConceptTableReadError):
        _hr(_source(root), [1, 2])


def test_a_new_batch_reads_the_new_bytes(tmp_path):
    root = _install(tmp_path / "eicu")
    _hr(_source(root), [2, 4])
    write_fixture_parquet(_patient((48, 40, 36, 12)), root / "patient.parquet", index=False)
    clear_stay_tables()  # the streamed batch boundary
    later = _hr(_source(root), [2, 4])
    clear_stay_tables()
    fresh = _hr(_source(_install(tmp_path / "fresh", (48, 40, 36, 12))), [2, 4])
    pd.testing.assert_frame_equal(later, fresh, check_exact=True)


def test_the_unscoped_memo_of_one_resolver_is_bound_too(tmp_path):
    root = _install(tmp_path / "eicu")
    source = _source(root)
    resolver = ConceptResolver(ConceptDictionary.from_json(DICTIONARY))
    resolver._stay_table(source, "patient")
    stat = (root / "patient.parquet").stat()
    os.utime(root / "patient.parquet", ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(StayTableSourceChanged):
        resolver._stay_table(source, "patient")


class _Files:
    """A source whose storage owner names ``files`` under one shared path."""

    def __init__(self, base: Path, files):
        self.config = type("Config", (), {"name": "eicu"})()
        self.base_path = base
        self._files = tuple(files)

    def resolve_target_files(self, table):
        return self._files


def test_a_source_over_the_same_path_naming_other_files_misses(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    a.write_bytes(b"a"), b.write_bytes(b"bb")
    reads = []

    def read(tag):
        def _read():
            reads.append(tag)
            return pd.DataFrame({"x": [tag]})
        return _read

    scope = ("patientunitstayid", frozenset({1}))
    first = stay_table_frame(source_binding(_Files(tmp_path, [a]), "patient"), scope, read("a"))
    second = stay_table_frame(source_binding(_Files(tmp_path, [b]), "patient"), scope, read("b"))
    assert reads == ["a", "b"]
    assert first["x"].tolist() == ["a"] and second["x"].tolist() == ["b"]


def test_a_validated_file_that_is_gone_is_a_typed_failure(tmp_path):
    gone = tmp_path / "patient.parquet"
    with pytest.raises(StayTableSourceChanged) as error:
        source_binding(_Files(tmp_path, [gone]), "patient")
    assert error.value.reason == "stay_table_source_missing"
    assert isinstance(error.value, StorageIntegrityError)


def test_a_source_without_files_shares_nothing(tmp_path):
    one, two = _Files(tmp_path, []), _Files(tmp_path, [])
    assert source_binding(one, "patient") != source_binding(two, "patient")
    assert isinstance(source_binding(one, "patient"), SourceBinding)
