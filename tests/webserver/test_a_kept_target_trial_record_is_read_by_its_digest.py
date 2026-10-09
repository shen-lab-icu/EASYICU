"""A kept target trial record is read back by the digest that names it.

A study's section names the record the host compiled for its card by the
record's digest; the host keeps the record, with the population it compiled,
in a file that digest names under its own state.  A record is written once:
keeping it again is a no-op, and other bytes under its digest are refused.
A read checks the record against its digest and its file, so a record
missing, edited, unreadable or filed under another digest is refused with
its own code.  The file names no study.  Synthetic records only.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from easyicu.webserver import target_trial_records as records
from tests.support.target_trial import (
    STUDY_ID,
    compiled_target_trial,
    kept_target_trial_record,
    target_trial_spec,
)


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "records"
    monkeypatch.setattr(records, "records_root", lambda: root)
    return root


def _refused(study_id: str, digest: str) -> records.TargetTrialRecordError:
    with pytest.raises(records.TargetTrialRecordError) as caught:
        records.load_target_trial_record(study_id, digest)
    return caught.value


def test_a_record_is_kept_once_and_read_back(_isolated_root: Path) -> None:
    kept = kept_target_trial_record()

    records.keep_target_trial_record(STUDY_ID, kept)
    records.keep_target_trial_record(STUDY_ID, kept)

    assert records.load_target_trial_record(STUDY_ID, kept.compile_sha256) == kept
    (path,) = _isolated_root.rglob("*.json")
    assert path.name == f"{kept.compile_sha256}.json"
    assert STUDY_ID not in str(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # Kept for one study, it is not another study's.
    assert _refused("study_other0001", kept.compile_sha256).code == (
        "target_trial_record_missing"
    )


def test_other_bytes_under_a_kept_digest_are_refused(_isolated_root: Path) -> None:
    kept = kept_target_trial_record()
    records.keep_target_trial_record(STUDY_ID, kept)
    (path,) = _isolated_root.rglob("*.json")
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(records.TargetTrialRecordError) as caught:
        records.keep_target_trial_record(STUDY_ID, kept)

    assert caught.value.code == "target_trial_record_identity_drift"


@pytest.mark.parametrize("damage", ["edited", "not_json", "other_digest", "too_large"])
def test_a_damaged_record_is_refused(_isolated_root: Path, damage: str) -> None:
    kept = kept_target_trial_record()
    records.keep_target_trial_record(STUDY_ID, kept)
    (path,) = _isolated_root.rglob("*.json")
    entry = json.loads(path.read_text(encoding="utf-8"))
    digest = kept.compile_sha256
    if damage == "edited":
        entry["record"]["protocol"] = entry["record"]["protocol"][:-1]
        path.write_text(json.dumps(entry), encoding="utf-8")
    elif damage == "not_json":
        path.write_bytes(b"\xff not a record")
    elif damage == "other_digest":
        # A whole record, filed under the digest of another.
        other = kept_target_trial_record(
            compiled=compiled_target_trial(
                spec=target_trial_spec(strategies={"initiate_label": "Prompt start"})
            )
        )
        digest = other.compile_sha256
        path.rename(path.with_name(f"{digest}.json"))
    else:
        path.write_bytes(b" " * (records.MAX_RECORD_BYTES + 1))

    assert _refused(STUDY_ID, digest).code == "target_trial_record_invalid"


def test_a_record_is_named_by_its_study_and_a_digest() -> None:
    for study_id, digest in (("", "a" * 64), (STUDY_ID, "A" * 64), (STUDY_ID, "a" * 63)):
        assert _refused(study_id, digest).code == (
            "target_trial_record_coordinates_invalid"
        )
    assert _refused(STUDY_ID, "a" * 64).code == "target_trial_record_missing"
