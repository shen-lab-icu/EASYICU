"""Partition invariance for SOFA: chunk size must not change the score.

Chunk size is an execution parameter. The same cohort, the same data and the
same concept definition must produce byte-identical scores whether the run is
chunked or not — otherwise a reported SOFA depends on how much memory the host
happened to have, or on which batch flag the operator typed.

MEASURED 2026-07-25 against real prepared data at
``/Volumes/外置硬盘/databases``. ``sofa`` and ``sofa2`` came out byte-identical
for every combination below:

===============  ==========  =========================================
database         cohort      result
===============  ==========  =========================================
MIMIC-IV         1,000       identical for chunk 250 / 500 / 1000
MIMIC-IV         3,000       identical for chunk 250…4000, workers 1/4
MIMIC-IV         10,000      identical for chunk 500 / 2000 / 4000
eICU             3,000       identical for chunk 500 / 2000
===============  ==========  =========================================

That measurement replaced an older unverified note in ``api.py`` claiming chunk
size could change large-cohort window expansion. Full-database scale (~94k
stays) is still unmeasured.

Run it with a real prepared database on hand::

    EASYICU_DATA_PATH=/Volumes/外置硬盘/databases/mimiciv \\
      pytest tests/core/test_sofa_partition_invariance.py --run-real
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

import easyicu

CHUNK_SIZES = [None, 250, 500, 1000, 2000, 4000]

#: Concepts whose windows straddle chunk boundaries in different ways.
SCORE_CONCEPTS = ["sofa", "sofa2"]


def _data_path() -> str:
    path = os.environ.get("EASYICU_DATA_PATH", "")
    if not path:
        pytest.skip("EASYICU_DATA_PATH is not set")
    return path


def _sorted_for_compare(frame: pd.DataFrame) -> pd.DataFrame:
    """Order rows and columns so only *content* differences can fail."""

    ordered = frame.reindex(sorted(frame.columns), axis=1)
    return ordered.sort_values(list(ordered.columns), kind="mergesort").reset_index(
        drop=True
    )


def _load(concept: str, *, chunk_size, patients: int, database: str):
    kwargs = {
        "database": database,
        "data_path": _data_path(),
        "max_patients": patients,
        "sample_strategy": "sorted",
        "verbose": False,
    }
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    return easyicu.load_concepts(concept, **kwargs)


@pytest.mark.needs_real_data
@pytest.mark.parametrize("concept", SCORE_CONCEPTS)
@pytest.mark.parametrize("chunk_size", [c for c in CHUNK_SIZES if c is not None])
def test_sofa_is_invariant_across_chunk_sizes(concept, chunk_size):
    """Chunked and unchunked runs must agree exactly, not approximately."""

    database = os.environ.get("EASYICU_TEST_DATABASE", "miiv")
    patients = int(os.environ.get("EASYICU_TEST_COHORT", "3000"))

    reference = _load(concept, chunk_size=None, patients=patients, database=database)
    chunked = _load(
        concept, chunk_size=chunk_size, patients=patients, database=database
    )

    pd.testing.assert_frame_equal(
        _sorted_for_compare(reference),
        _sorted_for_compare(chunked),
        check_exact=True,
        obj=f"{concept} @ chunk_size={chunk_size}",
    )


@pytest.mark.needs_real_data
@pytest.mark.parametrize("concept", SCORE_CONCEPTS)
def test_sofa_is_invariant_across_worker_counts(concept):
    """Parallelism must not reorder its way into a different score either."""

    database = os.environ.get("EASYICU_TEST_DATABASE", "miiv")
    patients = int(os.environ.get("EASYICU_TEST_COHORT", "3000"))
    common = {
        "database": database,
        "data_path": _data_path(),
        "max_patients": patients,
        "sample_strategy": "sorted",
        "verbose": False,
    }

    single = easyicu.load_concepts(concept, parallel_workers=1, **common)
    parallel = easyicu.load_concepts(concept, parallel_workers=4, **common)

    pd.testing.assert_frame_equal(
        _sorted_for_compare(single),
        _sorted_for_compare(parallel),
        check_exact=True,
        obj=f"{concept} @ parallel_workers",
    )


@pytest.mark.needs_real_data
def test_chunk_boundary_cases_are_represented_in_the_cohort():
    """Guard the guard: a cohort that never straddles a boundary proves nothing.

    A pass on a cohort smaller than one chunk would be vacuous, so assert the
    cohort is large enough for the smallest chunk size under test to split it.
    """

    database = os.environ.get("EASYICU_TEST_DATABASE", "miiv")
    patients = int(os.environ.get("EASYICU_TEST_COHORT", "3000"))
    smallest = min(c for c in CHUNK_SIZES if c is not None)

    assert patients > smallest * 2, (
        f"EASYICU_TEST_COHORT={patients} is too small to exercise chunking at "
        f"chunk_size={smallest}; the invariance tests would pass vacuously"
    )
