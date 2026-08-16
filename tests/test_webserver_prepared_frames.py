"""Characterization of how a prepared module frame gets read off disk.

Cohort Statistics and Patient Review each carried their own copy of this
stack — ``_MODULE_COLUMNS``, ``_read_selected_columns``, ``_read_module_frame``
and ``_fallback_entity_frame`` — inside two modules of 3211 and 2615 lines. The
column allow-lists are genuinely per-consumer (Cohort Statistics tolerates many
outcome-column spellings across export vintages; Patient Review reads a fixed
set), but the *mechanics* underneath them are one responsibility with two
implementations, and almost nothing covered them.

These tests pin the behaviour that exists today so the extraction can be shown
not to change it. They are written against the public surfaces of both callers
and must keep passing, unedited, after the owner is introduced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from easyicu.webserver import cohort_review
from easyicu.webserver import patient_drilldown


pd = pytest.importorskip("pandas")


def _frame() -> Any:
    return pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "death": [0, 1, 0, 1],
            "los_icu": [2.5, 8.0, 1.25, 30.0],
            "unwanted": ["a", "b", "c", "d"],
        }
    )


def _desc(file_name: str, rows: int = 4) -> Dict[str, Any]:
    return {
        "files": [
            {
                "module": "outcome",
                "file": file_name,
                "rows": rows,
                "columns": ["stay_id", "death", "los_icu", "unwanted"],
            }
        ]
    }


def _write(tmp_path: Path, suffix: str) -> str:
    name = f"outcome{suffix}"
    target = tmp_path / name
    frame = _frame()
    if suffix == ".parquet":
        pytest.importorskip("pyarrow")
        frame.to_parquet(target, index=False)
    elif suffix == ".xlsx":
        pytest.importorskip("openpyxl")
        frame.to_excel(target, index=False)
    else:
        frame.to_csv(target, index=False)
    return name


@pytest.mark.parametrize("suffix", [".parquet", ".csv", ".xlsx"])
def test_only_the_requested_columns_are_read(tmp_path: Path, suffix: str) -> None:
    """Every format dispatches to a reader that projects columns."""

    name = _write(tmp_path, suffix)
    frame = cohort_review._read_selected_columns(tmp_path / name, ["stay_id", "death"])

    assert list(frame.columns) == ["stay_id", "death"]
    assert len(frame) == 4
    assert "unwanted" not in frame.columns


@pytest.mark.parametrize("suffix", [".parquet", ".csv", ".xlsx"])
def test_patient_review_pushes_the_entity_filter_into_the_read(
    tmp_path: Path, suffix: str
) -> None:
    """The same projection, plus a row predicate the cohort reader lacks."""

    name = _write(tmp_path, suffix)
    frame = patient_drilldown._read_selected_columns(
        tmp_path / name,
        ["stay_id", "death"],
        stay_ids={"2", "4"},
        entity_column="stay_id",
    )

    assert list(frame.columns) == ["stay_id", "death"]
    assert sorted(str(value) for value in frame["stay_id"]) == ["2", "4"]


def test_without_entity_ids_both_readers_agree(tmp_path: Path) -> None:
    """`stay_ids=None` is the cohort reader's behaviour exactly."""

    name = _write(tmp_path, ".parquet")
    columns = ["stay_id", "death", "los_icu"]
    left = cohort_review._read_selected_columns(tmp_path / name, columns)
    right = patient_drilldown._read_selected_columns(tmp_path / name, columns)

    assert list(left.columns) == list(right.columns)
    assert left.to_dict("records") == right.to_dict("records")


def test_an_unknown_entity_id_yields_no_rows_not_every_row(tmp_path: Path) -> None:
    name = _write(tmp_path, ".parquet")
    frame = patient_drilldown._read_selected_columns(
        tmp_path / name, ["stay_id", "death"], stay_ids={"999"}, entity_column="stay_id"
    )
    assert len(frame) == 0


def test_module_frame_resolves_the_file_from_the_descriptor(tmp_path: Path) -> None:
    name = _write(tmp_path, ".parquet")
    frame = cohort_review._read_module_frame(tmp_path, _desc(name), "outcome")

    assert frame is not None
    assert "stay_id" in frame.columns
    # The descriptor lists a column the module allow-list does not want.
    assert "unwanted" not in frame.columns


def test_module_frame_is_none_when_the_descriptor_lacks_the_module(
    tmp_path: Path,
) -> None:
    name = _write(tmp_path, ".parquet")
    desc = _desc(name)
    assert cohort_review._read_module_frame(tmp_path, desc, "demographics") is None
    assert patient_drilldown._read_module_frame(tmp_path, desc, "demographics") is None


def test_module_frame_is_none_without_a_resolvable_entity_column(
    tmp_path: Path,
) -> None:
    name = _write(tmp_path, ".parquet")
    desc = _desc(name)
    desc["files"][0]["columns"] = ["death", "los_icu"]

    assert cohort_review._read_module_frame(tmp_path, desc, "outcome") is None
    assert patient_drilldown._read_module_frame(tmp_path, desc, "outcome") is None


def test_cohort_skips_a_huge_time_indexed_module_that_is_not_parquet(
    tmp_path: Path,
) -> None:
    """A guard Patient Review does not have, and must keep not having.

    Cohort Statistics reads whole modules; a multi-million-row score table in
    CSV would be read end to end for a summary. Parquet is exempt because the
    reader projects columns without materialising the rest.
    """

    csv_name = _write(tmp_path, ".csv")
    desc = _desc(csv_name, rows=cohort_review._INTERACTIVE_TIME_INDEXED_READ_ROW_LIMIT + 1)
    desc["files"][0]["module"] = "sofa2_score"

    assert "sofa2_score" in cohort_review._INTERACTIVE_SKIP_MODULES
    assert cohort_review._read_module_frame(tmp_path, desc, "sofa2_score") is None

    # Same row count as parquet: read, because the projection is cheap there.
    parquet_name = _write(tmp_path, ".parquet")
    desc["files"][0]["file"] = parquet_name
    assert cohort_review._read_module_frame(tmp_path, desc, "sofa2_score") is not None

    # Under the limit the CSV is read too.
    desc["files"][0]["file"] = csv_name
    desc["files"][0]["rows"] = 10
    assert cohort_review._read_module_frame(tmp_path, desc, "sofa2_score") is not None


def test_a_non_canonical_entity_column_is_renamed_to_stay_id(tmp_path: Path) -> None:
    """eICU-style exports key on patientunitstayid; readers canonicalize it."""

    frame = pd.DataFrame({"patientunitstayid": [7, 8], "death": [0, 1]})
    pytest.importorskip("pyarrow")
    frame.to_parquet(tmp_path / "outcome.parquet", index=False)
    desc: Dict[str, Any] = {
        "files": [
            {
                "module": "outcome",
                "file": "outcome.parquet",
                "rows": 2,
                "columns": ["patientunitstayid", "death"],
            }
        ]
    }

    for reader in (cohort_review._read_module_frame, patient_drilldown._read_module_frame):
        result = reader(tmp_path, desc, "outcome")
        assert result is not None
        assert "stay_id" in result.columns
        assert "patientunitstayid" not in result.columns


def test_fallback_entity_frame_returns_one_id_column(tmp_path: Path) -> None:
    name = _write(tmp_path, ".parquet")
    desc = _desc(name)

    for fallback in (
        cohort_review._fallback_entity_frame,
        patient_drilldown._fallback_entity_frame,
    ):
        frame = fallback(tmp_path, desc)
        assert frame is not None
        assert list(frame.columns) == ["stay_id"]
        assert len(frame) == 4


def test_fallback_entity_frame_is_none_without_any_id_column(tmp_path: Path) -> None:
    name = _write(tmp_path, ".parquet")
    desc = _desc(name)
    desc["files"][0]["columns"] = ["death"]

    assert cohort_review._fallback_entity_frame(tmp_path, desc) is None
    assert patient_drilldown._fallback_entity_frame(tmp_path, desc) is None


PARQUET_ID_DTYPES: List[Any] = ["int64", "float64", "string"]


@pytest.mark.parametrize("dtype", PARQUET_ID_DTYPES)
def test_parquet_pushdown_matches_the_stored_id_dtype(tmp_path: Path, dtype: str) -> None:
    """Predicate pushdown compares against the parquet schema's own type.

    A string predicate against an int64 column silently matches nothing, which
    would look like an empty cohort rather than an error.
    """

    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"stay_id": pd.Series([1, 2, 3], dtype=dtype), "death": [0, 1, 0]})
    target = tmp_path / "outcome.parquet"
    frame.to_parquet(target, index=False)

    result = patient_drilldown._read_selected_columns(
        target, ["stay_id", "death"], stay_ids={"2"}, entity_column="stay_id"
    )
    assert len(result) == 1
    assert str(result["stay_id"].iloc[0]).startswith("2")


# Every module that still dispatches on file suffix to a pandas reader with a
# column projection. This is a ratchet measured on 2026-08-16, not an
# endorsement: the two review surfaces were consolidated because they read the
# *same* prepared module frames for the same entity contract. The rest are
# genuinely different surfaces — a catalog preview, the Cross-DB aggregate
# path, Idea Mining's sampling reader, and the converter itself — each with its
# own extra parameters (`nrows`, `skiprows`, `max_records`). Folding them in
# needs its own characterization pass per surface, so they are recorded here
# rather than quietly ignored. The set may shrink; it may not grow.
KNOWN_UNCONSOLIDATED_READERS = {
    "catalog.py",
    "crossdb_review.py",
    "dataio.py",
    "ideas/mining.py",
    "patient_drilldown/__init__.py",  # row-paged table preview, not module frames
}

WEBSERVER = Path(cohort_review.__file__).resolve().parent


def _suffix_dispatching_readers() -> set[str]:
    """Files that fork on file format to pick a pandas reader.

    Detected by the xlsx/csv fork rather than by `read_parquet`: the row-paged
    preview in patient_drilldown reads parquet by iterating row groups, so a
    `read_parquet(` probe would have missed a real copy.
    """

    found = set()
    for path in sorted(WEBSERVER.rglob("*.py")):
        if path.name == "prepared_frames.py":
            continue
        source = path.read_text(encoding="utf-8")
        if "read_excel(" in source and "read_csv(" in source:
            found.add(path.relative_to(WEBSERVER).as_posix())
    return found


def test_the_review_surfaces_no_longer_carry_their_own_reader() -> None:
    """Cohort Statistics and Patient Review read the same prepared frames.

    They had two copies of the mechanism, and only one of them pushed the
    entity predicate into the parquet read. That is the duplication this owner
    removes; the ratchet below records what is left.
    """

    readers = _suffix_dispatching_readers()
    assert "cohort_review.py" not in readers, (
        "cohort_review.py re-implements the prepared-frame reader; "
        "call prepared_frames.read_selected_columns instead"
    )


def test_no_new_module_starts_dispatching_on_file_suffix() -> None:
    readers = _suffix_dispatching_readers()
    new = readers - KNOWN_UNCONSOLIDATED_READERS
    assert new == set(), (
        "these modules grew their own prepared-file reader: "
        f"{sorted(new)} — use webserver/prepared_frames.py"
    )
    gone = KNOWN_UNCONSOLIDATED_READERS - readers
    assert gone == set(), (
        f"consolidated already — drop from the ratchet: {sorted(gone)}"
    )
