import gzip
import hashlib
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.acquisition.eicu_first_observed_candidate import (
    FLAG_COLUMN,
    REQUIRED_COLUMNS,
    FirstObservedCandidateError,
)
from easyicu.webserver import eicu_first_observed_source as owner


@pytest.fixture
def source(tmp_path, monkeypatch):
    rows = pd.DataFrame([
        (1, "a", 1, 1, "MICU", "35"),
        (2, "b", 2, 1, "SICU", "75"),
    ], columns=REQUIRED_COLUMNS)
    rows["hospitaldischargestatus"] = ["Expired", "Alive"]
    patient = tmp_path / "patient.parquet"
    rows.to_parquet(patient, index=False)
    archive = tmp_path / "official.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("demo/patient.csv.gz", gzip.compress(rows.to_csv(index=False).encode()))
    source = SimpleNamespace(database="eicu_demo", archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                             max_uncompressed_bytes=1_000_000)
    binding = SimpleNamespace(
        family="eicu", database="eicu_demo", table_paths={"patient": patient},
        table_sha256={"patient": hashlib.sha256(patient.read_bytes()).hexdigest()},
        export_manifest_sha256="a" * 64,
    )
    monkeypatch.setattr(owner, "resolve_raw_source_identity_binding", lambda **_: binding)
    monkeypatch.setattr(owner, "verified_public_demo_raw", lambda _: source)
    monkeypatch.setattr(owner, "source_paths", lambda _: SimpleNamespace(archive=archive))
    return rows, patient, archive, binding


def test_complete_source_compares_identity_and_age_without_outcome(source, monkeypatch):
    read = owner.pd.read_csv
    seen = []

    def recording_read(*args, **kwargs):
        result = read(*args, **kwargs)
        seen.extend(result.columns)
        return result

    monkeypatch.setattr(owner.pd, "read_csv", recording_read)
    result = owner.derive_from_verified_eicu_demo(source[1].parent)
    assert result.frame[FLAG_COLUMN].sum() == 2
    assert set(seen) == set(REQUIRED_COLUMNS)
    proof = result.receipt["source_binding"]["complete_patient_source"]
    assert proof["all_source_rows_compared"] is True
    assert proof["outcome_columns_read"] is False
    assert proof["research_execution_authorized"] is False
    assert result.receipt["public_projection_authorized"] is False
    assert '"a"' not in str(proof)


@pytest.mark.parametrize("change", ["truncate", "patient", "age", "archive"])
def test_refuses_incomplete_mismatched_or_unpinned_inputs(source, change):
    rows, patient, archive, binding = source
    if change == "truncate":
        rows = rows.iloc[:1]
    elif change == "patient":
        rows.loc[0, "uniquepid"] = "different"
    elif change == "age":
        rows.loc[0, "age"] = "36"
    else:
        archive.write_bytes(archive.read_bytes() + b"changed")
    if change != "archive":
        rows.to_parquet(patient, index=False)
        binding.table_sha256["patient"] = hashlib.sha256(patient.read_bytes()).hexdigest()
    with pytest.raises(FirstObservedCandidateError):
        owner.derive_from_verified_eicu_demo(patient.parent)


def test_unverified_source_is_refused(source, monkeypatch):
    _, patient, _, _ = source
    monkeypatch.setattr(owner, "verified_public_demo_raw", lambda _: None)
    with pytest.raises(FirstObservedCandidateError, match="not verified"):
        owner.derive_from_verified_eicu_demo(patient.parent)


def test_ambiguous_patient_members_are_refused(source):
    rows, patient, archive, _ = source
    spec = owner.verified_public_demo_raw(patient.parent)
    content = gzip.compress(rows.to_csv(index=False).encode())
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("one/patient.csv.gz", content)
        zipped.writestr("two/patient.csv.gz", content)
    spec.archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(FirstObservedCandidateError, match="ambiguous"):
        owner.derive_from_verified_eicu_demo(patient.parent)
