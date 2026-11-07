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


# --- every reader family refuses a marked install (r13slpub increment) -----------------


T = pd.Timestamp("2150-01-01 08:00")


def _policy_source(database: str, root: Path) -> ICUDataSource:
    from easyicu.resources import load_data_sources

    return ICUDataSource(config=load_data_sources().get(database), base_path=root)


def _stays(ids):
    return pd.DataFrame(
        {
            "row_id": ids,
            "subject_id": [i * 10 for i in ids],
            "hadm_id": [i * 100 for i in ids],
            "icustay_id": ids,
            "intime": [T] * len(ids),
            "outtime": [T + pd.Timedelta(days=2)] * len(ids),
        }
    )


def _csv_stays(install: Path) -> Path:
    install.mkdir(parents=True, exist_ok=True)
    path = install / "icustays.csv"
    _stays([1, 2]).to_csv(path, index=False)
    return path


def _refused(call) -> None:
    with pytest.raises(SealPublicationError) as caught:
        call()
    assert caught.value.reason in {
        "bucket_install_unpublished",
        "bucket_install_refused",
    }


def test_csv_admission_refuses_an_unmanifested_marked_install(tmp_path):
    from easyicu.storage_layout import csv_table_scan

    install = tmp_path / "mimic_r13"
    path = _csv_stays(install)
    _mark(install, ".seal_PENDING")
    _refused(lambda: csv_table_scan(_policy_source("mimic", install), "icustays", path))


def test_csv_admission_refuses_a_marked_install_under_the_unregistered_override(
    tmp_path, monkeypatch
):
    from easyicu.storage_layout import ALLOW_UNREGISTERED_SCAN_ENV, csv_table_scan

    install = tmp_path / "mimic_r13"
    path = _csv_stays(install)
    (install / "conversion_manifest.json").write_text(json.dumps({"tables": []}))
    monkeypatch.setenv(ALLOW_UNREGISTERED_SCAN_ENV, "1")
    csv_table_scan(_policy_source("mimic", install), "icustays", path)
    _mark(install, ".seal_PENDING")
    _refused(lambda: csv_table_scan(_policy_source("mimic", install), "icustays", path))


def _csv_services(install: Path) -> None:
    # MIMIC-III services is the table the disk loader reads from its CSV.
    install.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": [1],
            "subject_id": [10],
            "hadm_id": [100],
            "transfertime": [T],
            "prev_service": [None],
            "curr_service": ["MED"],
        }
    ).to_csv(install / "services.csv", index=False)
    # load_table fills services' stay column from icustays.
    _stays([1000]).assign(subject_id=10, hadm_id=100).to_parquet(
        install / "icustays.parquet"
    )


def test_load_table_refuses_the_csv_of_a_marked_install(tmp_path):
    install = tmp_path / "mimic_r13"
    _csv_services(install)
    assert len(_policy_source("mimic", install).load_table("services").data) == 1
    _mark(install, ".seal_PENDING")
    _refused(lambda: _policy_source("mimic", install).load_table("services"))


def test_the_fast_id_scan_refuses_the_csv_of_a_marked_install(tmp_path):
    from types import SimpleNamespace

    from easyicu.api.concepts import _build_fast_scan_expr

    install = tmp_path / "mimic_r13"
    _csv_services(install)
    loader = SimpleNamespace(datasource=_policy_source("mimic", install))
    assert _build_fast_scan_expr(loader, "services")
    _mark(install, ".seal_PENDING")
    loader = SimpleNamespace(datasource=_policy_source("mimic", install))
    _refused(lambda: _build_fast_scan_expr(loader, "services"))


def test_patient_id_discovery_refuses_the_csv_of_a_marked_install(tmp_path):
    from easyicu.api import get_all_patient_ids

    install = tmp_path / "mimic_r13"
    _csv_stays(install)
    ids, _column = get_all_patient_ids(install, database="mimic")
    assert sorted(ids) == [1, 2]
    _mark(install, ".seal_PENDING")
    _refused(lambda: get_all_patient_ids(install, database="mimic"))


def test_the_patient_filter_refuses_the_csv_of_a_marked_install(tmp_path):
    from easyicu.patient_filter import PatientFilter

    install = tmp_path / "mimic_r13"
    _csv_stays(install)
    path = install / "icustays.csv"
    assert len(PatientFilter("mimic", install)._read_csv("icustays", path)) == 2
    _mark(install, ".seal_PENDING")
    # _read_table already refuses through resolve_target_files (016ddb655);
    # the CSV admission itself (patient_filter._read_csv) must refuse too.
    _refused(lambda: PatientFilter("mimic", install)._read_table("icustays"))
    _refused(lambda: PatientFilter("mimic", install)._read_csv("icustays", path))


def test_hirid_end_time_refuses_the_observation_parts_of_a_marked_install(tmp_path):
    from easyicu.concept import ConceptDictionary, ConceptResolver

    install = tmp_path / "hirid_r13"
    (install / "observations").mkdir(parents=True)
    for part, pid in ((1, 1), (2, 2)):
        pd.DataFrame(
            {"patientid": [pid], "datetime": [T], "variableid": [200]}
        ).to_parquet(install / "observations" / f"{part}.parquet")
    (install / "observation_tables").mkdir()
    pd.DataFrame({"patientid": [1, 2], "part": [0, 1]}).to_csv(
        install / "observation_tables" / "observation_tables_index.csv", index=False
    )
    frame = pd.DataFrame({"patientid": [1, 2]})

    def synthesise():
        return ConceptResolver(ConceptDictionary({}))._synthesise_los_column(
            "end", _policy_source("hirid", install), frame
        )

    assert synthesise() is not None
    _mark(install, ".seal_PENDING")
    _refused(synthesise)


def test_an_absent_table_of_a_marked_install_raises_instead_of_reading_empty(tmp_path):
    from easyicu.storage_layout import verify_flat_scan

    install = tmp_path / "miiv_r13"
    install.mkdir()
    source = _policy_source("miiv", install)
    assert source.load_table("chartevents").data.empty
    assert verify_flat_scan(source, "chartevents", ()) == ()
    _mark(install, ".seal_PENDING")
    _refused(lambda: _policy_source("miiv", install).load_table("chartevents"))
    _refused(
        lambda: verify_flat_scan(_policy_source("miiv", install), "chartevents", ())
    )


def test_the_eicu_io_reader_refuses_a_directory_of_a_marked_install(tmp_path):
    from easyicu.concept.eicu_io import read_io_parquet

    install = tmp_path / "eicu_r13"
    (install / "intakeoutput").mkdir(parents=True)
    pd.DataFrame(
        {"patientunitstayid": [1], "intakeoutputoffset": [30], "cellpath": ["x"]}
    ).to_parquet(install / "intakeoutput" / "1.parquet")
    assert len(read_io_parquet(install / "intakeoutput", None, None, None)) == 1
    _mark(install, ".seal_PENDING")
    _refused(lambda: read_io_parquet(install / "intakeoutput", None, None, None))


def test_the_mimic_column_peek_refuses_a_directory_of_a_marked_install(tmp_path):
    from easyicu.datasource import mimic_disk_columns

    install = tmp_path / "miiv_r13"
    (install / "labevents").mkdir(parents=True)
    pd.DataFrame(
        {"itemid": [50912], "valuenum": [1.0], "valueuom": ["mg/dL"]}
    ).to_parquet(install / "labevents" / "1.parquet")
    source = _policy_source("miiv", install)
    assert "valueuom" in mimic_disk_columns(
        source, install / "labevents", "labevents", ["itemid"]
    )
    _mark(install, ".seal_PENDING")
    _refused(
        lambda: mimic_disk_columns(
            source, install / "labevents", "labevents", ["itemid"]
        )
    )


def test_a_cached_table_is_not_served_after_the_install_is_refused(tmp_path):
    install = tmp_path / "mimic_r13"
    _csv_services(install)
    source = _policy_source("mimic", install)
    assert len(source.load_table("services").data) == 1
    _mark(install, ".seal_REFUSED")
    _refused(lambda: source.load_table("services"))


def test_the_id_map_search_does_not_skip_a_refused_table():
    from types import SimpleNamespace

    from easyicu.io.data_load import load_id_map

    class Marked:
        def load_table(self, table, columns=None):
            raise SealPublicationError(
                "bucket_install_unpublished", Path("/x/miiv_r13")
            )

    config = SimpleNamespace(
        name="miiv",
        id_configs={
            "icustay": SimpleNamespace(table="icustays", position=3),
            "hadm": SimpleNamespace(table="admissions", position=2),
        },
    )
    with pytest.raises(SealPublicationError):
        load_id_map(Marked(), config, "hadm_id", "stay_id")
