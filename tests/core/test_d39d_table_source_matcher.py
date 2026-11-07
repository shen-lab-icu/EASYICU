"""One matcher of a table's on-disk entries, shared by stay discovery and the reader (delta39d).

B-D39-PRESENCE-CASE: presence tried only the configured, lower and upper
case. On a case-sensitive filesystem a mixed-case ``IcuStays.parquet``
(healthy or dangling), an empty ``IcuStays/`` or a ``.PARQUET`` suffix was
therefore "absent", and stay discovery read the CSV. The normal loader's
case-insensitive lookup did find the file, so the reader and the cohort
saw different rows. ``storage_layout.match_table_entries`` now lists the
actual entries, casefold-matched, under every contract root, and the
reader's loader is reconciled with the same matcher.

B-D39-PRESENCE-CONFLICT: the first readable candidate won. Two independent
same-name sources (case variants, base and ``icu``, two nested directories,
links to different directories) are now ``table_source_ambiguous``; the
only exemption is a flat source bound by its bucket's ``derived_from``.
Aliases of one physical source (one realpath, or a hardlink) count once. A
broken entry beside a valid one is broken. Rows are synthetic.

Case-variant cases need a case-sensitive filesystem; they are skipped
elsewhere (this matrix is run on the Linux server).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest

from easyicu import storage_layout as sl
from easyicu.config import load_src_cfg
from easyicu.datasource import ICUDataSource
from easyicu.storage_layout import StorageIntegrityError

AMBIGUOUS = "table_source_ambiguous"
UNREADABLE = "table_scan_file_unreadable"
BROKEN = "table_layout_broken"
CSV_IDS = [900, 901]
IDS = list(range(100, 120))


def _case_sensitive(tmp_path) -> bool:
    probe = tmp_path / "case_probe"
    probe.mkdir()
    (probe / "lower").write_text("x")
    return not (probe / "LOWER").exists()


@pytest.fixture
def case_sensitive(tmp_path):
    if not _case_sensitive(tmp_path):
        pytest.skip("needs a case-sensitive filesystem")


def _root(tmp_path, name="mimic"):
    root = tmp_path / name
    root.mkdir()
    pd.DataFrame({"icustay_id": CSV_IDS}).to_csv(root / "icustays.csv", index=False)
    return root


def _parquet(path, ids=IDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"icustay_id": list(ids), "value": np.repeat(60.0, len(ids))}).to_parquet(path, index=False)


def _dangling(path, root):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(root / "nowhere")


def _discover(root):
    from easyicu.api import get_all_patient_ids

    listing: dict = {}
    ids, _column = get_all_patient_ids(root, database="mimic", listing=listing)
    return ids, listing


def _refused(root):
    with pytest.raises(StorageIntegrityError) as error:
        _discover(root)
    return getattr(error.value, "reason", "")


# --- B-D39-PRESENCE-CASE ------------------------------------------------------

@pytest.mark.parametrize("relative", ["IcuStays.parquet", "IcuStays/part.parquet", "icustays.PARQUET"])
def test_a_healthy_table_in_another_spelling_is_the_stay_table(tmp_path, case_sensitive, relative):
    root = _root(tmp_path)
    _parquet(root / relative)
    ids, listing = _discover(root)
    assert ids == IDS and listing["source_format"] == "parquet"


@pytest.mark.parametrize("make,reason", [
    (lambda r: _dangling(r / "IcuStays.parquet", r), UNREADABLE),
    (lambda r: (r / "IcuStays").mkdir(), BROKEN),
    (lambda r: (r / "IcuStays_bucket").mkdir(), BROKEN),
    (lambda r: _dangling(r / "ICU", r), UNREADABLE),
    (lambda r: _dangling(r / "icustays.PARQUET", r), UNREADABLE),
    (lambda r: _dangling(r / "custom" / "IcuStays.parquet", r), UNREADABLE),
], ids=["mixed_flat_link", "mixed_empty_dir", "mixed_empty_bucket", "mixed_icu_root",
        "upper_suffix_link", "nested_mixed_link"])
def test_a_broken_table_in_another_spelling_is_refused(tmp_path, case_sensitive, make, reason):
    root = _root(tmp_path)
    make(root)
    assert _refused(root) == reason


def test_mixed_case_normal_reader_vs_cohort(tmp_path, case_sensitive):
    """MIXED_CASE_NORMAL_READER_VS_COHORT: the reader and the cohort read one source."""
    root = _root(tmp_path)
    _parquet(root / "IcuStays.parquet")
    source = ICUDataSource(load_src_cfg("mimic"), base_path=root, enable_cache=False)
    loader = source._resolve_loader_from_disk("icustays")
    read = pd.concat(
        [pd.read_parquet(f) for f in source.resolve_target_files("icustays", loader=loader)],
        ignore_index=True,
    )
    ids, listing = _discover(root)
    assert ids == read["icustay_id"].tolist() == IDS and listing["source_format"] == "parquet"


# --- B-D39-PRESENCE-CONFLICT --------------------------------------------------

def test_two_spellings_of_two_sources_are_ambiguous(tmp_path, case_sensitive):
    root = _root(tmp_path)
    _parquet(root / "icustays.parquet")
    _parquet(root / "ICUSTAYS.parquet", range(200, 220))
    assert _refused(root) == AMBIGUOUS


def test_conflicting_symlink_directories(tmp_path, case_sensitive):
    """CONFLICTING_SYMLINK_DIRECTORIES: two spellings linked to different sources."""
    root = _root(tmp_path)
    _parquet(root / "left" / "part.parquet")
    _parquet(root / "right" / "part.parquet", range(200, 220))
    (root / "icustays").symlink_to(root / "left", target_is_directory=True)
    (root / "ICUSTAYS").symlink_to(root / "right", target_is_directory=True)
    assert _refused(root) == AMBIGUOUS


def test_a_broken_spelling_beside_a_valid_one_is_broken(tmp_path, case_sensitive):
    root = _root(tmp_path)
    _parquet(root / "icustays.parquet")
    (root / "ICUSTAYS.parquet").mkdir()
    assert _refused(root) == BROKEN


@pytest.mark.parametrize("layout", ["base_and_icu", "two_nested", "dir_links_base_and_icu"])
def test_independent_sources_at_two_places_are_ambiguous(tmp_path, layout):
    root = _root(tmp_path)
    if layout == "base_and_icu":
        _parquet(root / "icustays.parquet")
        _parquet(root / "icu" / "icustays.parquet", range(200, 220))
    elif layout == "two_nested":
        _parquet(root / "left" / "icustays.parquet")
        _parquet(root / "right" / "icustays.parquet", range(200, 220))
    else:
        _parquet(root / "a" / "part.parquet")
        _parquet(root / "b" / "part.parquet", range(200, 220))
        (root / "icustays").symlink_to(root / "a", target_is_directory=True)
        (root / "icu").mkdir()
        (root / "icu" / "icustays").symlink_to(root / "b", target_is_directory=True)
    assert _refused(root) == AMBIGUOUS


def test_the_reader_refuses_the_same_ambiguity(tmp_path):
    root = _root(tmp_path)
    _parquet(root / "icustays.parquet")
    _parquet(root / "icu" / "icustays.parquet", range(200, 220))
    source = ICUDataSource(load_src_cfg("mimic"), base_path=root, enable_cache=False)
    with pytest.raises(StorageIntegrityError) as error:
        source._resolve_loader_from_disk("icustays")
    assert error.value.reason == AMBIGUOUS


def test_a_broken_entry_beside_a_valid_one_is_broken(tmp_path):
    root = _root(tmp_path)
    _parquet(root / "icustays.parquet")
    (root / "icu").mkdir()
    (root / "icu" / "icustays").mkdir()
    assert _refused(root) == BROKEN


@pytest.mark.parametrize("alias", ["hardlink", "same_directory_link"])
def test_aliases_of_one_physical_source_count_once(tmp_path, alias):
    root = _root(tmp_path)
    if alias == "hardlink":
        _parquet(root / "icustays.parquet")
        (root / "icu").mkdir()
        os.link(root / "icustays.parquet", root / "icu" / "icustays.parquet")
    else:
        _parquet(root / "backing" / "part.parquet")
        (root / "icustays").symlink_to(root / "backing", target_is_directory=True)
        (root / "icu").mkdir()
        (root / "icu" / "icustays").symlink_to(root / "backing", target_is_directory=True)
    ids, listing = _discover(root)
    assert ids == IDS and listing["source_format"] == "parquet"


def _bucket(root, *, derived: bool):
    """A range bucket of icustays beside its flat source, bound by derived_from or not."""
    flat = root / "icustays.parquet"
    frame = pd.read_parquet(flat).assign(**{sl.ORDINAL: np.arange(len(IDS), dtype="int64")})
    bucket = root / "icustays_bucket"
    (bucket / "bucket_id=0").mkdir(parents=True)
    (bucket / "_COMPLETE").write_text("synthetic")
    (bucket / sl.BUILD_RECEIPT).write_text("{}")
    part = bucket / "bucket_id=0" / "part.parquet"
    frame.to_parquet(part, index=False)
    manifest = {
        "schema": sl.SCHEMA, "scheme": "range", "key_column": "icustay_id", "key_type": "BIGINT",
        "range_breakpoints": [], "sort_keys": ["icustay_id", sl.ORDINAL],
        "source_order": sl.ORDINAL, "row_group_target": 65536,
        "builder_version": sl.BUILDER_VERSION,
        "inventory": [{"relative_path": "bucket_id=0/part.parquet",
                       "size": part.stat().st_size, "row_count": len(IDS)}],
    }
    if derived:
        stat = flat.stat()
        manifest["derived_from"] = {"table": "icustays", "files": [{
            "path": "icustays.parquet", "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": hashlib.sha256(flat.read_bytes()).hexdigest(),
        }]}
    (bucket / sl.MANIFEST).write_text(json.dumps(manifest))


def test_a_flat_source_and_its_derived_bucket_are_one_source(tmp_path):
    root = _root(tmp_path)
    _parquet(root / "icustays.parquet")
    _bucket(root, derived=True)
    ids, listing = _discover(root)
    assert ids == IDS and listing["source_format"] == "parquet"


def test_a_flat_source_and_an_unbound_bucket_are_ambiguous(tmp_path):
    root = _root(tmp_path)
    _parquet(root / "icustays.parquet")
    _bucket(root, derived=False)
    assert _refused(root) == AMBIGUOUS


def test_an_absent_table_still_uses_the_csv(tmp_path):
    ids, listing = _discover(_root(tmp_path))
    assert sorted(ids) == CSV_IDS and listing["source_format"] == "csv_fallback"
