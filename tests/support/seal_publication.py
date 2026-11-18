"""Synthetic installations and receipts for storage seal publication tests."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from easyicu.storage_layout import MANIFEST
from easyicu.storage_seal_publication import IN_ROOT_RECEIPT
from tests.support.bucket_storage import write_contract


def publication_bucket(install: Path, table: str = "chartevents") -> Path:
    root = install / f"{table}_bucket"
    (root / "bucket_id=0").mkdir(parents=True)
    pd.DataFrame({"itemid": [220045], "valuenum": [71.0]}).to_parquet(
        root / "bucket_id=0" / "1.parquet"
    )
    write_contract(root, key="itemid")
    return root


def publication_receipt(install: Path, **fields) -> dict:
    receipt = {"schema": "lead_r13.seal_build_receipt/2", "verdict": "PASS"}
    receipt.update(fields)
    (install / IN_ROOT_RECEIPT).write_text(json.dumps(receipt))
    return receipt


def sealed_install(install: Path, table: str = "chartevents") -> Path:
    root = publication_bucket(install, table)
    publication_receipt(
        install,
        tables={table: {"manifest_sha256": hashlib.sha256((root / MANIFEST).read_bytes()).hexdigest(), "qc_findings": []}},
    )
    return root
