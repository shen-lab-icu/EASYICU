"""Path-dependent dependency and source failures are typed (Codex delta2 item 11).

- Sepsis-3 with published score modules and no batch size raised a plain
  ValueError although the derivation does not depend on the batch size;
- a missing streamed dependency raised a plain FileNotFoundError, and an
  unreadable dependency manifest looked like a missing one;
- the converter skipped a source directory it could not list (os.walk's
  default) and declared "no converted output" for an unreadable folder.
Synthetic files only.
"""

from __future__ import annotations

import os

import pytest

import easyicu.api.extraction as extraction
from easyicu.api.extraction import SpecialDependencyError


def _published(root, modules=("sepsis_shared", "sofa1_score")):
    root.mkdir(parents=True, exist_ok=True)
    for module in modules:
        (root / f"{module}.parquet").write_bytes(b"")
    return root


def test_published_sepsis_dependencies_without_a_batch_derive_in_owner_batches(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        extraction, "_stream_special_extraction_batches",
        lambda modules, database, data_path, ids, batch, out, **kw: calls.append(batch),
    )
    published = _published(tmp_path / "published")
    cohort = {"stay_id": list(range(1, 51))}
    extraction._run_special_extraction(
        ["sepsis3_sofa1"], "miiv", str(tmp_path), cohort, None, str(tmp_path / "out"),
        published_output_dir=str(published),
    )
    expected = extraction._special_dependency_batch_policy(
        "miiv", ["sepsis_shared", "sofa1_score"], 50, None
    )["derivation_batch_size"]
    assert calls == [expected]

    # An explicit batch is used as given.
    extraction._run_special_extraction(
        ["sepsis3_sofa1"], "miiv", str(tmp_path), cohort, 7, str(tmp_path / "out"),
        published_output_dir=str(published),
    )
    assert calls[-1] == 7


def test_published_sepsis_dependencies_without_a_cohort_are_a_typed_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction, "_stream_special_extraction_batches", lambda *a, **k: None)
    published = _published(tmp_path / "published")
    with pytest.raises(SpecialDependencyError) as caught:
        extraction._run_special_extraction(
            ["sepsis3_sofa1"], "miiv", str(tmp_path), None, None, str(tmp_path / "out"),
            published_output_dir=str(published),
        )
    assert caught.value.reason_code == "sepsis_special_cohort_missing"


def _stream(tmp_path, published):
    extraction._stream_special_extraction_batches(
        ["sepsis3_sofa1"], "miiv", str(tmp_path), {"stay_id": [1, 2]}, 10,
        str(tmp_path / "out"), use_sofa2=False, published_output_dir=str(published),
    )


def test_a_missing_streamed_dependency_is_typed(tmp_path):
    published = tmp_path / "published"
    published.mkdir()
    (tmp_path / "out").mkdir()
    with pytest.raises(SpecialDependencyError) as caught:
        _stream(tmp_path, published)
    assert caught.value.reason_code == "sepsis_dependency_output_missing"
    assert not isinstance(caught.value, FileNotFoundError)


def test_an_unreadable_dependency_manifest_is_not_a_missing_one(tmp_path):
    published = tmp_path / "published"
    published.mkdir()
    (tmp_path / "out").mkdir()
    (published / "sepsis_shared.manifest.json").write_text("{not json")
    with pytest.raises(SpecialDependencyError) as caught:
        _stream(tmp_path, published)
    assert caught.value.reason_code == "sepsis_dependency_manifest_unreadable"


def test_an_unreadable_published_manifest_is_not_rederived(tmp_path):
    published = tmp_path / "published"
    published.mkdir()
    (published / "sepsis_shared.manifest.json").write_text("{not json")
    produced = []
    with pytest.raises(SpecialDependencyError) as caught:
        extraction._stage_special_dependencies(
            ["sepsis_shared"], staging_dir=tmp_path / "staging", published_output_dir=str(published),
            group_output_root=None, produce=lambda module, target: produced.append(module),
        )
    assert caught.value.reason_code == "sepsis_dependency_manifest_unreadable"
    assert produced == []


# --- converter --------------------------------------------------------------

needs_posix_permissions = pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="needs POSIX permissions and a non-root user",
)


def _converter(path, database="miiv"):
    from easyicu.io.data_converter import DataConverter

    return DataConverter(path, database=database, verbose=False)


@needs_posix_permissions
def test_an_unlistable_source_directory_stops_the_scan(tmp_path):
    from easyicu.io.data_converter import ConversionSourceUnreadable

    (tmp_path / "icu").mkdir()
    (tmp_path / "icu" / "chartevents.csv").write_text("a\n1\n")
    (tmp_path / "hosp").mkdir()
    (tmp_path / "hosp" / "labevents.csv").write_text("a\n1\n")
    os.chmod(tmp_path / "hosp", 0)
    try:
        with pytest.raises(ConversionSourceUnreadable) as caught:
            _converter(tmp_path)._get_csv_files()
    finally:
        os.chmod(tmp_path / "hosp", 0o700)
    assert caught.value.reason == "conversion_source_unreadable"
    assert caught.value.path.name == "hosp"
    # Readable again: both tables are found.
    names = sorted(p.name for p in _converter(tmp_path)._get_csv_files())
    assert names == ["chartevents.csv", "labevents.csv"]


def test_hidden_system_folders_are_not_sources(tmp_path):
    (tmp_path / ".Trashes").mkdir()
    (tmp_path / ".Trashes" / "old.csv").write_text("a\n1\n")
    (tmp_path / "icu").mkdir()
    (tmp_path / "icu" / "chartevents.csv").write_text("a\n1\n")
    assert [p.name for p in _converter(tmp_path)._get_csv_files()] == ["chartevents.csv"]


@needs_posix_permissions
def test_an_unlistable_output_folder_is_not_no_converted_output(tmp_path):
    from easyicu.io.data_converter import ConversionSourceUnreadable

    (tmp_path / "chartevents").mkdir()
    (tmp_path / "chartevents" / "1.parquet").write_bytes(b"PAR1")
    os.chmod(tmp_path / "chartevents", 0)
    try:
        with pytest.raises(ConversionSourceUnreadable):
            _converter(tmp_path)._has_converted_outputs()
    finally:
        os.chmod(tmp_path / "chartevents", 0o700)
    assert _converter(tmp_path)._has_converted_outputs() is True
