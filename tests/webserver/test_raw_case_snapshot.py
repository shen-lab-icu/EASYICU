"""Synthetic-only checks: a staged snapshot does not grant data access."""
from pathlib import Path

import pytest

from easyicu.research_agent.authority.filesystem import AnchoredDirectory, AuthorityFilesystemError
from easyicu.webserver.raw_case_snapshot import stage_reviewed_source_files, verify_staged_source_files


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "cases.parquet").write_bytes(b"synthetic case bytes")
    (source / "laboratory").mkdir()
    (source / "laboratory" / "1.parquet").write_bytes(b"synthetic measurement bytes")
    return source


def test_stage_and_verify_are_explicitly_not_a_producer_receipt(tmp_path):
    source, stage = _source(tmp_path), tmp_path / "stage"
    receipt = stage_reviewed_source_files(
        source_root=source, staging_root=stage,
        relative_files=["laboratory/1.parquet", "cases.parquet"], source_identity_sha256="a" * 64,
    )
    assert [x.relative_path for x in receipt.files] == ["cases.parquet", "laboratory/1.parquet"]
    assert (stage / "cases.parquet").read_bytes() == b"synthetic case bytes"
    public = receipt.public_receipt()
    assert public["source_identity_verified"] is False
    assert public["point_in_time_source_snapshot_verified"] is False
    assert public["actual_consumption_verified"] is False
    assert public["read_roster_complete"] is False
    verify_staged_source_files(stage, receipt)
    (stage / "cases.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="staged_bytes_changed"):
        verify_staged_source_files(stage, receipt)


@pytest.mark.parametrize("relative", ["../outside", ".", "./cases.parquet", "/outside", "a\\b"])
def test_reject_unsafe_names_before_copy(tmp_path, relative):
    source, stage = _source(tmp_path), tmp_path / "stage"
    with pytest.raises(ValueError, match="invalid_relative_path"):
        stage_reviewed_source_files(source_root=source, staging_root=stage,
                                    relative_files=[relative], source_identity_sha256="a" * 64)
    assert not stage.exists()


def test_source_change_during_copy_does_not_yield_a_receipt(tmp_path, monkeypatch):
    source, stage = _source(tmp_path), tmp_path / "stage"
    original_open = AnchoredDirectory.open_regular
    source_opens = 0

    def changed_on_second_open(directory, name):
        nonlocal source_opens
        if directory.path == source and name == "cases.parquet":
            source_opens += 1
            if source_opens == 2:
                (source / name).write_bytes(b"replaced during staging")
        return original_open(directory, name)

    monkeypatch.setattr(AnchoredDirectory, "open_regular", changed_on_second_open)
    with pytest.raises(ValueError, match="source_changed"):
        stage_reviewed_source_files(source_root=source, staging_root=stage,
                                    relative_files=["cases.parquet"], source_identity_sha256="a" * 64)
    assert not (stage / "cases.parquet").exists()


def test_reject_symlinks_and_existing_snapshot(tmp_path):
    source, stage = _source(tmp_path), tmp_path / "stage"
    (source / "alias.parquet").symlink_to(source / "cases.parquet")
    with pytest.raises(AuthorityFilesystemError):
        stage_reviewed_source_files(source_root=source, staging_root=stage,
                                    relative_files=["alias.parquet"], source_identity_sha256="a" * 64)
    with pytest.raises(FileExistsError):
        stage_reviewed_source_files(source_root=source, staging_root=stage,
                                    relative_files=["cases.parquet"], source_identity_sha256="a" * 64)


def test_concurrent_target_is_never_overwritten(tmp_path, monkeypatch):
    source, stage = _source(tmp_path), tmp_path / "stage"
    original_publish = AnchoredDirectory.publish_temporary_no_clobber

    def race(directory, temporary, target):
        (directory.path / target).write_bytes(b"concurrent owner data")
        return original_publish(directory, temporary, target)

    monkeypatch.setattr(AnchoredDirectory, "publish_temporary_no_clobber", race)
    with pytest.raises(FileExistsError):
        stage_reviewed_source_files(source_root=source, staging_root=stage,
                                    relative_files=["cases.parquet"], source_identity_sha256="a" * 64)
    assert (stage / "cases.parquet").read_bytes() == b"concurrent owner data"


def test_real_sic_case_export_can_consume_a_staged_synthetic_source(tmp_path):
    import pandas as pd
    from easyicu.research_agent.intake.export_package import open_export_package
    from easyicu.webserver.dataio import make_export_runner
    from easyicu.webserver.time_origin_source_authority import case_duration_source_provenance
    from tests.support.data_provenance import ExportJob

    source, stage, out = tmp_path / "source", tmp_path / "stage", tmp_path / "export"
    source.mkdir()
    pd.DataFrame({
        "CaseID": [1, 2], "ICUOffset": [0, 3600], "TimeOfStay": [172800, 176400],
        "OffsetOfDeath": [None, None], "HospitalDischargeType": [2026, 2026],
    }).to_parquet(source / "cases.parquet", index=False)
    pd.DataFrame({
        "CaseID": [1, 1, 2], "Offset": [3600, 86400, 7200],
        "LaboratoryID": [657, 657, 657], "LaboratoryValue": [1.0, 9.0, 2.0],
    }).to_parquet(source / "laboratory.parquet", index=False)
    receipt = stage_reviewed_source_files(
        source_root=source, staging_root=stage, relative_files=["cases.parquet", "laboratory.parquet"],
        source_identity_sha256="a" * 64,
    )
    # Synthetic original becomes unavailable: actual exporter must use the
    # staged source. This is not a proof for other rosters or production runs.
    (source / "cases.parquet").unlink()
    (source / "laboratory.parquet").unlink()
    result = make_export_runner(
        data_path=str(stage), database="sic", modules=["blood_gas", "sic_case_landmark"],
        concepts={"blood_gas": ["lact"], "sic_case_landmark": [
            "care_unit_to_case_closure_days", "death_during_case", "death_time_during_case_hours",
        ]},
        export_format="parquet", out_dir=str(out), include_feature_definitions=False,
    )(ExportJob())
    assert result["file_count"] == 2
    columns = pd.read_parquet(out / "sic_case_landmark.parquet")
    assert columns["care_unit_to_case_closure_days"].tolist() == [2.0, 2.0]
    assert columns["death_during_case"].tolist() == [False, False]
    lactate = pd.read_parquet(out / "blood_gas.parquet")
    assert lactate["charttime"].tolist() == [1.0, 24.0, 1.0]
    assert lactate["lact"].tolist() == [1.0, 9.0, 2.0]
    with open_export_package(out) as package:
        assert package.declared_time_origin == "first_intensive_or_intermediate_care_transfer"
        assert case_duration_source_provenance(package).implementation_sha256
    verify_staged_source_files(stage, receipt)
    assert receipt.public_receipt()["actual_consumption_verified"] is False
