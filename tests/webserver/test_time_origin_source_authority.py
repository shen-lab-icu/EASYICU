"""The registered export receipt is not a scientific approval."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.research_agent.research_context.export_selection import SourceIdentity
from easyicu.webserver import time_origin_source_authority as authority


def test_exact_registered_package_has_content_bound_receipt(tmp_path, monkeypatch):
    identity = SourceIdentity(database="sic", registration_id="source", version="1.0.6")
    package = SimpleNamespace(root=tmp_path, database="sic", declared_time_origin=SIC_CARE_UNIT_ORIGIN,
                              authority_sha256="a" * 64, manifest_sha256="b" * 64)
    monkeypatch.setattr(authority.sources, "registered_source_identity", lambda _: identity)
    monkeypatch.setattr(authority, "open_export_package", lambda _: nullcontext(package))
    receipt = authority.attest_registered_export_time_origin(tmp_path, expected_identity=identity)
    assert receipt.source_identity_sha256 == canonical_sha256(identity.to_dict())
    assert receipt.export_authority_sha256 == "a" * 64
    assert receipt.manifest_sha256 == "b" * 64
    assert receipt.selected_origin == SIC_CARE_UNIT_ORIGIN


@pytest.mark.parametrize("mismatch", ["source", "registration_changed", "package", "legacy_origin", "path"])
def test_receipt_rejects_missing_or_changed_authority(tmp_path, monkeypatch, mismatch):
    identity = SourceIdentity(database="sic", registration_id="source")
    other = SourceIdentity(database="sic", registration_id="another")
    package = SimpleNamespace(root=tmp_path, database="sic", declared_time_origin=SIC_CARE_UNIT_ORIGIN,
                              authority_sha256="a" * 64, manifest_sha256="b" * 64)
    reads = iter([identity, other] if mismatch == "registration_changed" else [other, other] if mismatch == "source" else [identity, identity])
    monkeypatch.setattr(authority.sources, "registered_source_identity", lambda _: next(reads))
    if mismatch == "package":
        package.database = "miiv"
    if mismatch == "legacy_origin":
        package.declared_time_origin = "icu_admission"
    if mismatch == "path":
        package.root = tmp_path.parent
    monkeypatch.setattr(authority, "open_export_package", lambda _: nullcontext(package))
    with pytest.raises(ValueError):
        authority.attest_registered_export_time_origin(tmp_path, expected_identity=identity)


def test_case_duration_requires_one_complete_recorded_source_snapshot():
    from easyicu.concept.sic_landmark import SIC_CASE_LANDMARK_COLUMNS, SIC_CASE_SOURCE_COLUMNS

    sources = tuple(SimpleNamespace(table="cases", value_variable=name, callback="sic_case_landmark_fields")
                    for name in SIC_CASE_SOURCE_COLUMNS)
    concepts = tuple(SimpleNamespace(concept_id=name, status="recorded", sources=sources)
                     for name in SIC_CASE_LANDMARK_COLUMNS)
    snapshot = SimpleNamespace(source_database="sic", source_config_sha256="a" * 64,
                               implementation_sha256="b" * 64, snapshot_sha256="c" * 64, concepts=concepts)
    package = SimpleNamespace(data_provenance=SimpleNamespace(snapshots=(snapshot,)),
                              concept_index={name: {} for name in SIC_CASE_LANDMARK_COLUMNS})
    result = authority.case_duration_source_provenance(package)
    assert result.snapshot_sha256 == "c" * 64
    snapshot.concepts = concepts[:-1]
    with pytest.raises(ValueError, match="case_duration_source_provenance_unavailable"):
        authority.case_duration_source_provenance(package)
