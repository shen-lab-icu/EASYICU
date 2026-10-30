"""The approved snapshot must contain no additional readable entries."""
import os

import pytest

from easyicu.research_agent.authority.filesystem import AnchoredDirectory, AuthorityFilesystemError
from easyicu.webserver.raw_case_snapshot import stage_reviewed_source_files, verify_staged_source_files


@pytest.mark.parametrize("extra", ["extra_file", "extra_dir", "extra_link", "case_variant", "parent_extra"])
def test_reject_extra_snapshot_entries(tmp_path, extra):
    source, stage = tmp_path / "source", tmp_path / "stage"
    (source / "laboratory").mkdir(parents=True)
    (source / "laboratory" / "lactate.parquet").write_bytes(b"synthetic")
    receipt = stage_reviewed_source_files(
        source_root=source, staging_root=stage,
        relative_files=["laboratory/lactate.parquet"], source_identity_sha256="a" * 64,
    )
    if extra == "extra_file":
        (stage / "laboratory" / "other.parquet").write_bytes(b"unapproved")
    elif extra == "extra_dir":
        (stage / "laboratory" / "other").mkdir()
    elif extra == "extra_link":
        (stage / "laboratory" / "other").symlink_to(source / "laboratory" / "lactate.parquet")
    elif extra == "case_variant":
        (stage / "laboratory" / "lactate.parquet").rename(stage / "laboratory" / "LACTATE.parquet")
    else:
        (stage / "unapproved").mkdir()
    with pytest.raises((ValueError, AuthorityFilesystemError), match="raw_snapshot_roster_changed"):
        verify_staged_source_files(stage, receipt)


def test_reject_replaced_expected_directory_symlink(tmp_path):
    source, stage = tmp_path / "source", tmp_path / "stage"
    (source / "part").mkdir(parents=True)
    (source / "part" / "data").write_bytes(b"synthetic")
    receipt = stage_reviewed_source_files(source_root=source, staging_root=stage,
                                          relative_files=["part/data"], source_identity_sha256="a" * 64)
    os.rename(stage / "part", stage / "old_part")
    (stage / "part").symlink_to(source / "part", target_is_directory=True)
    with pytest.raises((ValueError, AuthorityFilesystemError)):
        verify_staged_source_files(stage, receipt)


def test_reject_persistent_change_to_earlier_source_during_later_copy(tmp_path, monkeypatch):
    source, stage = tmp_path / "source", tmp_path / "stage"
    source.mkdir()
    (source / "a").write_bytes(b"first-version")
    (source / "b").write_bytes(b"second-file")
    publish = AnchoredDirectory.publish_temporary_no_clobber

    def mutate_after_second_copy(self, temporary, name):
        result = publish(self, temporary, name)
        if name == "b":
            (source / "a").write_bytes(b"new-version")
        return result

    monkeypatch.setattr(AnchoredDirectory, "publish_temporary_no_clobber", mutate_after_second_copy)
    with pytest.raises(ValueError, match="raw_snapshot_source_changed"):
        stage_reviewed_source_files(source_root=source, staging_root=stage,
                                    relative_files=["a", "b"], source_identity_sha256="a" * 64)
