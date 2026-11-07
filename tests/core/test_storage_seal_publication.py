"""Readers refuse a bucket installation the seal tool has not published (r13slpub).

Synthetic fixtures only. The seal tool's names (``.<install>.seal_PENDING``,
``.<install>.seal_REFUSED``, ``_SEAL_BUILD_RECEIPT.json``) are those of
tools/seal_buckets/seal_rebuild.py at review-r13sealtool d09b1fcb3.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from easyicu.config import DataSourceConfig
from easyicu.datasource import ICUDataSource
from easyicu.storage_layout import (
    MANIFEST,
    BucketLayoutError,
    StorageIntegrityError,
    read_bucket_contract,
    resolve_table_files,
)
from easyicu.storage_seal_publication import (
    IN_ROOT_RECEIPT,
    SealPublicationError,
    assert_install_published,
    assert_seal_published,
)
from tests.support.bucket_storage import write_contract


def _bucket(install: Path, table: str = "chartevents") -> Path:
    root = install / f"{table}_bucket"
    (root / "bucket_id=0").mkdir(parents=True)
    pd.DataFrame({"itemid": [220045], "valuenum": [71.0]}).to_parquet(
        root / "bucket_id=0" / "1.parquet"
    )
    write_contract(root, key="itemid")
    return root


def _flat(install: Path, table: str = "chartevents") -> Path:
    flat = install / table
    flat.mkdir(parents=True)
    pd.DataFrame({"itemid": [220045], "valuenum": [71.0]}).to_parquet(
        flat / "1.parquet"
    )
    return flat


def _mark(install: Path, suffix: str) -> Path:
    marker = install.parent / f".{install.name}{suffix}"
    marker.write_text(json.dumps({"root": str(install), "verdict": suffix[6:]}))
    return marker


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(install: Path, **fields) -> dict:
    receipt = {"schema": "lead_r13.seal_build_receipt/2", "verdict": "PASS"}
    receipt.update(fields)
    (install / IN_ROOT_RECEIPT).write_text(json.dumps(receipt))
    return receipt


def _sealed(install: Path, table: str = "chartevents") -> Path:
    root = _bucket(install, table)
    _receipt(
        install,
        tables={table: {"manifest_sha256": _sha(root / MANIFEST), "qc_findings": []}},
    )
    return root


def _source(install: Path) -> ICUDataSource:
    return ICUDataSource(DataSourceConfig(name="miiv"), base_path=install)


def _reason(excinfo) -> str:
    return getattr(excinfo.value, "reason", None)


# --- read time: markers -----------------------------------------------------------


def test_a_pending_bucket_install_is_refused_at_read_time(tmp_path):
    install = tmp_path / "sic_r13"
    root = _bucket(install)
    read_bucket_contract(root)
    _mark(install, ".seal_PENDING")
    with pytest.raises(StorageIntegrityError) as caught:
        read_bucket_contract(root)
    assert _reason(caught) == "bucket_install_unpublished"
    # Not a layout error: the bucket resolver must not fall back to flat files.
    assert not isinstance(caught.value, BucketLayoutError)


def test_a_pending_install_does_not_fall_back_to_its_flat_files(tmp_path):
    install = tmp_path / "miiv_r13"
    _bucket(install)
    _flat(install)
    _mark(install, ".seal_PENDING")
    with pytest.raises(StorageIntegrityError) as caught:
        _source(install).resolve_bucket_directory("chartevents")
    assert _reason(caught) == "bucket_install_unpublished"


def test_a_pending_flat_install_is_refused_at_read_time(tmp_path):
    install = tmp_path / "miiv_r13"
    _flat(install)
    _mark(install, ".seal_PENDING")
    with pytest.raises(StorageIntegrityError) as caught:
        resolve_table_files(_source(install), "chartevents")
    assert _reason(caught) == "bucket_install_unpublished"


def test_a_refused_install_is_refused_for_a_bucket_and_a_flat_read(tmp_path):
    install = tmp_path / "eicu_r13"
    root = _bucket(install)
    _flat(install, "vitalperiodic")
    _mark(install, ".seal_REFUSED")
    with pytest.raises(StorageIntegrityError) as caught:
        read_bucket_contract(root)
    assert _reason(caught) == "bucket_install_refused"
    with pytest.raises(StorageIntegrityError) as caught:
        resolve_table_files(_source(install), "vitalperiodic")
    assert _reason(caught) == "bucket_install_refused"


def test_a_marker_beside_the_realpath_of_a_symlinked_install_is_refused(tmp_path):
    real = tmp_path / "b6_root" / "sic_r13"
    root = _bucket(real)
    alias = tmp_path / "data" / "sic"
    alias.parent.mkdir()
    alias.symlink_to(real, target_is_directory=True)
    read_bucket_contract(alias / root.name)
    _mark(real, ".seal_PENDING")
    with pytest.raises(StorageIntegrityError) as caught:
        read_bucket_contract(alias / root.name)
    assert _reason(caught) == "bucket_install_unpublished"
    with pytest.raises(StorageIntegrityError) as caught:
        resolve_table_files(_source(alias), "chartevents")
    assert _reason(caught) == "bucket_install_unpublished"


def test_a_carevue_farm_link_to_a_pending_install_is_refused_at_read_time(tmp_path):
    b6 = tmp_path / "b6_root"
    sealed = b6 / "mimic_cv_r13"
    bucket = _bucket(sealed)
    farm = b6 / "mimic_cv_dir_r13"
    farm.mkdir()
    (farm / "chartevents_bucket").symlink_to(bucket, target_is_directory=True)
    read_bucket_contract(farm / "chartevents_bucket")
    _mark(sealed, ".seal_PENDING")
    with pytest.raises(StorageIntegrityError) as caught:
        read_bucket_contract(farm / "chartevents_bucket")
    assert _reason(caught) == "bucket_install_unpublished"


def test_a_dangling_marker_link_still_counts(tmp_path):
    install = tmp_path / "hirid_r13"
    root = _bucket(install)
    (tmp_path / ".hirid_r13.seal_PENDING").symlink_to(tmp_path / "gone")
    with pytest.raises(SealPublicationError) as caught:
        read_bucket_contract(root)
    assert _reason(caught) == "bucket_install_unpublished"


def test_an_unmarked_legacy_install_reads_as_before(tmp_path):
    install = tmp_path / "miiv_flat"
    root = _bucket(install)
    _flat(install, "labevents")
    assert read_bucket_contract(root).files
    source = _source(install)
    assert source.resolve_bucket_directory("chartevents") == root
    assert resolve_table_files(source, "chartevents")
    # Non-formal: no receipt is required.
    assert_seal_published(root, formal=False)
    assert_install_published(install, formal=False)


# --- formal half (called by the formal run receipt, not by the reader) ---------------


def test_formal_accepts_a_bucket_its_receipt_sealed(tmp_path):
    install = tmp_path / "sic_r13"
    root = _sealed(install)
    assert_seal_published(root, formal=True)
    assert_install_published(install, formal=True)


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        {"verdict": "FAILED"},
        {"verdict": "PENDING"},
        {"verdict": "PASS", "failure": {"reason": "x"}},
        "not json",
    ],
    ids=["missing", "failed", "pending", "failure_entry", "unreadable"],
)
def test_formal_refuses_an_install_without_a_passing_receipt(tmp_path, receipt):
    install = tmp_path / "sic_r13"
    root = _sealed(install)
    path = install / IN_ROOT_RECEIPT
    if receipt is None:
        path.unlink()
    elif isinstance(receipt, str):
        path.write_text(receipt)
    else:
        sealed = json.loads(path.read_text())
        sealed.pop("verdict")
        path.write_text(json.dumps({**sealed, **receipt}))
    for check in (
        lambda: assert_seal_published(root, formal=True),
        lambda: assert_install_published(install, formal=True),
    ):
        with pytest.raises(SealPublicationError) as caught:
            check()
        assert _reason(caught) == "bucket_install_unaccepted"
    assert_seal_published(root, formal=False)


def test_formal_refuses_a_manifest_that_differs_from_the_receipt(tmp_path):
    install = tmp_path / "sic_r13"
    root = _sealed(install)
    manifest = json.loads((root / MANIFEST).read_text())
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1))
    with pytest.raises(SealPublicationError) as caught:
        assert_seal_published(root, formal=True)
    assert _reason(caught) == "bucket_install_unaccepted"


def test_formal_refuses_a_matching_sha_when_the_verdict_is_not_pass(tmp_path):
    install = tmp_path / "sic_r13"
    root = _sealed(install)
    receipt = json.loads((install / IN_ROOT_RECEIPT).read_text())
    assert receipt["tables"]["chartevents"]["manifest_sha256"] == _sha(root / MANIFEST)
    (install / IN_ROOT_RECEIPT).write_text(json.dumps({**receipt, "verdict": "FAILED"}))
    with pytest.raises(SealPublicationError) as caught:
        assert_seal_published(root, formal=True)
    assert _reason(caught) == "bucket_install_unaccepted"


def test_formal_refuses_a_table_record_with_findings(tmp_path):
    install = tmp_path / "sic_r13"
    root = _sealed(install)
    receipt = json.loads((install / IN_ROOT_RECEIPT).read_text())
    receipt["tables"]["chartevents"]["qc_findings"] = [
        "bucket_rows_inventory_vs_receipt"
    ]
    (install / IN_ROOT_RECEIPT).write_text(json.dumps(receipt))
    with pytest.raises(SealPublicationError):
        assert_seal_published(root, formal=True)


def test_formal_refuses_an_unsealed_bucket_and_a_flat_read_of_a_sealed_table(tmp_path):
    install = tmp_path / "sic_r13"
    _sealed(install)
    extra = _bucket(install, "laboratory")
    with pytest.raises(SealPublicationError) as caught:
        assert_seal_published(extra, formal=True)
    assert _reason(caught) == "bucket_install_unaccepted"
    other = tmp_path / "eicu_r13"
    _sealed(other)
    (other / "chartevents_bucket" / MANIFEST).unlink()
    with pytest.raises(SealPublicationError):
        assert_seal_published(other / "chartevents_bucket", formal=True)


def test_formal_follows_a_farm_link_to_the_realpath_receipt(tmp_path):
    b6 = tmp_path / "b6_root"
    sealed = b6 / "mimic_cv_r13"
    bucket = _sealed(sealed)
    farm = b6 / "mimic_cv_dir_r13"
    farm.mkdir()
    (farm / "chartevents_bucket").symlink_to(bucket, target_is_directory=True)
    _receipt(
        farm,
        schema="lead_r13.seal_farm_receipt/2",
        bucket_links={"chartevents_bucket": str(bucket)},
        bucket_root=str(sealed),
    )
    assert_seal_published(farm / "chartevents_bucket", formal=True)
    assert_install_published(farm, formal=True)
    # The realpath install's receipt binds the bucket: a mismatch there is refused.
    receipt = json.loads((sealed / IN_ROOT_RECEIPT).read_text())
    receipt["tables"]["chartevents"]["manifest_sha256"] = "0" * 64
    (sealed / IN_ROOT_RECEIPT).write_text(json.dumps(receipt))
    for check in (
        lambda: assert_seal_published(farm / "chartevents_bucket", formal=True),
        lambda: assert_install_published(farm, formal=True),
    ):
        with pytest.raises(SealPublicationError) as caught:
            check()
        assert _reason(caught) == "bucket_install_unaccepted"


def test_formal_never_accepts_a_farm_receipt_on_its_own(tmp_path):
    b6 = tmp_path / "b6_root"
    sealed = b6 / "mimic_cv_r13"
    bucket = _bucket(sealed)  # no receipt in the realpath install
    farm = b6 / "mimic_cv_dir_r13"
    farm.mkdir()
    (farm / "chartevents_bucket").symlink_to(bucket, target_is_directory=True)
    _flat(farm, "d_items")
    _receipt(farm, bucket_links={"chartevents_bucket": str(bucket)})
    for root in (farm / "chartevents_bucket", farm / "d_items_bucket"):
        with pytest.raises(SealPublicationError) as caught:
            assert_seal_published(root, formal=True)
        assert _reason(caught) == "bucket_install_unaccepted"
    _receipt(farm, bucket_links={})
    with pytest.raises(SealPublicationError):
        assert_install_published(farm, formal=True)


def test_formal_refuses_a_farm_link_that_points_elsewhere(tmp_path):
    b6 = tmp_path / "b6_root"
    sealed = b6 / "mimic_cv_r13"
    bucket = _sealed(sealed)
    stray = _sealed(b6 / "mimic_cv_old")
    farm = b6 / "mimic_cv_dir_r13"
    farm.mkdir()
    (farm / "chartevents_bucket").symlink_to(stray, target_is_directory=True)
    _receipt(farm, bucket_links={"chartevents_bucket": str(bucket)})
    with pytest.raises(SealPublicationError) as caught:
        assert_seal_published(farm / "chartevents_bucket", formal=True)
    assert _reason(caught) == "bucket_install_unaccepted"


def test_formal_still_refuses_a_marked_install_first(tmp_path):
    install = tmp_path / "sic_r13"
    root = _sealed(install)
    _mark(install, ".seal_PENDING")
    with pytest.raises(SealPublicationError) as caught:
        assert_seal_published(root, formal=True)
    assert _reason(caught) == "bucket_install_unpublished"
