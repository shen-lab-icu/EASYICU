"""Regression coverage for sealed v6's event and SOFA2 companion bindings."""
import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/figures/QC-A02_easyicu_cross_database_reliability_audit.py"
SPEC = importlib.util.spec_from_file_location("companion_qc", SCRIPT)
QC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QC)


def example(concept="death", companion="death_time", role="event_time"):
    entry = {"module": "outcome", "concept_ids": [concept],
             "column_metadata_columns": [concept, companion]}
    columns = {
        concept: {"metadata": {"column_name": concept, "source_concept": concept,
                               "role": "event_status" if concept == "death" else "value"}},
        companion: {"metadata": {"column_name": companion, "source_concept": concept, "role": role}},
    }
    sidecar = {"files": [{"module": "outcome", "relative_path": "outcome.parquet", "columns": columns}]}
    kwargs = {"sidecar": sidecar, "parquet_names": ["stay_id", "charttime", concept, companion],
              "sidecar_sha_matches": True}
    return entry, kwargs


@pytest.mark.parametrize("concept,companion,role", [
    ("death", "death_time", "event_time"),
    ("sofa2", "sofa2_observed", "measurement_status"),
    ("sofa2", "sofa2_available", "measurement_status"),
])
def test_sealed_primary_plus_typed_companion_is_complete(concept, companion, role):
    entry, kwargs = example(concept, companion, role)
    assert QC._concept_metadata_complete(entry, **kwargs)


@pytest.mark.parametrize("defect", [
    "missing_primary", "wrong_role", "wrong_source", "absent_column", "unverified_sidecar",
    "duplicate_binding", "duplicate_selection", "bad_column_name", "missing_binding",
    "wrong_file", "duplicate_module", "companion_as_primary", "unknown_extra",
])
def test_extra_column_names_do_not_bypass_metadata_contract(defect):
    entry, kwargs = example()
    binding = kwargs["sidecar"]["files"][0]
    columns = binding["columns"]
    if defect == "missing_primary":
        entry["column_metadata_columns"].remove("death")
        del columns["death"]
    elif defect == "wrong_role":
        columns["death_time"]["metadata"]["role"] = "value"
    elif defect == "wrong_source":
        columns["death_time"]["metadata"]["source_concept"] = "unselected"
    elif defect == "absent_column":
        kwargs["parquet_names"].remove("death_time")
    elif defect == "unverified_sidecar":
        kwargs["sidecar_sha_matches"] = False
    elif defect == "duplicate_binding":
        entry["column_metadata_columns"].append("death_time")
    elif defect == "duplicate_selection":
        entry["concept_ids"].append("death")
    elif defect == "bad_column_name":
        columns["death_time"]["metadata"]["column_name"] = "other"
    elif defect == "missing_binding":
        del columns["death_time"]
    elif defect == "wrong_file":
        binding["relative_path"] = "other.parquet"
    elif defect == "duplicate_module":
        kwargs["sidecar"]["files"].append(copy.deepcopy(binding))
    elif defect == "companion_as_primary":
        columns["death"]["metadata"]["role"] = "measurement_status"
    elif defect == "unknown_extra":
        entry["column_metadata_columns"].append("unbound")
    assert not QC._concept_metadata_complete(entry, **kwargs)
