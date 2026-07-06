from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_ROOT = Path(os.environ.get("EASYICU_DB_ROOT", "/Volumes/外置硬盘/databases"))


def _load_audit_module():
    path = REPO_ROOT / "tools" / "audit_full_concept_dictionary_structure.py"
    spec = importlib.util.spec_from_file_location("audit_full_concept_dictionary_structure", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not DEFAULT_DB_ROOT.exists(),
    reason="local ICU source dictionaries are not mounted",
)
def test_public_database_source_mappings_have_no_structural_errors() -> None:
    audit = _load_audit_module()
    schema = audit.load_data_source_schema()
    id_catalogs, eicu_labels = audit.build_catalogs(DEFAULT_DB_ROOT, scan_eicu_labels=False)

    rows = []
    for filename in audit.DICTIONARY_FILES:
        rows.extend(
            audit.audit_dictionary(
                dictionary_name=filename,
                payload=audit._read_json(audit.DATA_DIR / filename),
                schema=schema,
                id_catalogs=id_catalogs,
                eicu_labels=eicu_labels,
                dbs=set(audit.PUBLIC_DBS),
                include_demo=False,
            )
        )

    errors = [row for row in rows if row.status == "error"]
    assert errors == []
