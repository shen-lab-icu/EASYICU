"""R6 refined specimen order: each arterial gas hour publishes its specimen evidence.

Every clinical row is a synthetic literal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import easyicu
from easyicu.concept import ConceptDictionary, ConceptResolver
from easyicu.concept.loader import load_dictionary
from easyicu.concept_output_sources import SPECIMEN_STATUS_LEVELS, SPECIMEN_STATUS_OUTPUT_COLUMNS
from tests.support.blood_gas_sources import DICTIONARY, mimic_source
from tests.support.eicu_source import _eicu_source

ARTERIAL = ("po2", "pco2", "ph", "sao2", "be", "tco2")
CONCEPT_DICT = Path(easyicu.__file__).parent / "data" / "concept-dict.json"


def _eicu_panel(tmp_path):
    lab = pd.DataFrame([
        # hour 0: paO2 by name; pH beside it; a lone pH at 20 min has no anchor
        ("paO2", 80., 0), ("pH", 7.40, 0), ("pH", 7.30, 20),
        # hour 1: explicit arterial sample at 70, unrecorded at 80
        ("paO2", 90., 70), ("paO2", 60., 80), ("pH", 7.20, 80),
        # hour 2: venous sample type drops the whole panel
        ("paO2", 40., 130), ("pH", 7.35, 130),
    ], columns=["labname", "labresult", "labresultoffset"]).assign(labmeasurenameinterface="")
    sample = pd.DataFrame({"labothername": "Sample Type", "labotheroffset": [70, 130],
                           "labothervaluetext": ["Arterial", "Venous"], "labotherresult": ""})
    return _eicu_source(tmp_path, "eicu", lab=lab, customlab=sample)


def _status(table, concept):
    frame = table.data.sort_values(table.index_column)
    return dict(zip(frame[table.index_column], frame[f"{concept}_specimen_status"]))


def test_registry_covers_the_arterial_concepts_with_ordered_levels():
    assert set(SPECIMEN_STATUS_OUTPUT_COLUMNS) == set(ARTERIAL)
    assert SPECIMEN_STATUS_LEVELS == ("co_measured_arterial", "label_defined_arterial", "explicit_arterial")


def test_every_arterial_source_states_its_specimen_rule():
    """Non-eICU sources are either specimen-checked or arterial by item name."""
    dictionary = json.loads(CONCEPT_DICT.read_text())
    for concept in ARTERIAL:
        for database, sources in dictionary[concept]["sources"].items():
            for source in sources:
                callback = str(source.get("callback"))
                if database.startswith("eicu"):
                    assert "eicu_arterial_gas" in callback
                elif "arterial_specimen_only" in callback:
                    assert "specimen_status" not in source
                else:
                    assert source.get("specimen_status") == "label_defined_arterial", (concept, database)


def test_eicu_hour_publishes_its_weakest_admitted_evidence(tmp_path):
    loaded = ConceptResolver(load_dictionary()).load_concepts(
        ["po2", "ph"], _eicu_panel(tmp_path), merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), verbose=False, concept_workers=1,
    )
    assert _status(loaded["po2"], "po2") == {0.: "label_defined_arterial", 1.: "label_defined_arterial"}
    assert _status(loaded["ph"], "ph") == {0.: "co_measured_arterial", 1.: "co_measured_arterial"}
    ph = loaded["ph"].data.sort_values(loaded["ph"].index_column)
    assert ph.ph.tolist() == pytest.approx([7.40, 7.20])  # lone pH and venous panel excluded


@pytest.mark.parametrize("merge,r_compatible", [(True, False), (True, True)])
def test_merged_outputs_keep_the_status_beside_the_value(tmp_path, merge, r_compatible):
    out = ConceptResolver(load_dictionary()).load_concepts(
        ["po2", "ph"], _eicu_panel(tmp_path), merge=merge, r_compatible=r_compatible,
        interval=pd.Timedelta(hours=1), verbose=False, concept_workers=1,
    ).sort_values("charttime")
    assert out.po2_specimen_status.tolist() == ["label_defined_arterial"] * 2
    assert out.ph_specimen_status.tolist() == ["co_measured_arterial"] * 2


def test_explicit_specimen_records_in_mimic(tmp_path):
    source, id_column = mimic_source(tmp_path)
    loaded = ConceptResolver(ConceptDictionary.from_json(DICTIONARY)).load_concepts(
        ["po2"], source, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1),
        patient_ids={id_column: [1]}, concept_workers=1, verbose=False,
    )["po2"]
    # Only the gas with an ART specimen record is admitted.
    assert _status(loaded, "po2") == {0.: "explicit_arterial"}


@pytest.mark.parametrize("layout", ["flat", "bucket"])
def test_hirid_arterial_variables_are_label_defined(tmp_path, layout):
    from tests.support.hirid_source import hirid_source_root as _source_root
    from tests.support.blood_gas_sources import _source

    root = _source_root(tmp_path, "observations", [
        {"patientid": 1, "datetime": t, "variableid": item, "value": value}
        for t, item, value in [(0.1, 20000200, 80.), (0.5, 20000200, 90.), (1.2, 20000300, 7.4)]
    ], layout=layout)
    loaded = ConceptResolver(load_dictionary()).load_concepts(
        ["po2", "ph"], _source(root, "hirid"), merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), patient_ids={"patientid": [1]}, concept_workers=1, verbose=False,
    )
    assert _status(loaded["po2"], "po2") == {0.: "label_defined_arterial"}
    assert loaded["po2"].data.po2.tolist() == [85.]
    assert _status(loaded["ph"], "ph") == {1.: "label_defined_arterial"}


def test_native_export_types_the_status_as_categorical_text(tmp_path):
    from easyicu.api.extraction import _native_export_physical_value_columns, _native_export_storage_kind
    from easyicu.concept.export_metadata import build_export_file_metadata_binding

    assert "po2_specimen_status" in _native_export_physical_value_columns(["po2"])
    assert _native_export_storage_kind("po2_specimen_status", load_dictionary()) == "string"
    table = ConceptResolver(load_dictionary()).load_concepts(
        ["po2"], _eicu_panel(tmp_path), merge=False, r_compatible=False,
        interval=pd.Timedelta(hours=1), verbose=False, concept_workers=1,
    )["po2"]
    frame = table.data.rename(columns={table.index_column: "charttime", "patientunitstayid": "stay_id"})
    binding = build_export_file_metadata_binding(
        relative_path="blood_gas.parquet", module="blood_gas", frame=frame,
        concept_ids=["po2"], database="eicu", database_class_prefixes=(),
        dictionary=load_dictionary(),
    )
    column = binding.columns["po2_specimen_status"]
    metadata = column.metadata.to_dict()
    assert column.representation_transform == "specimen_status"
    assert metadata["source_concept"] == "po2"
    assert "mmHg" not in json.dumps(metadata.get("canonical_unit"))
    assert "label_defined_arterial" in metadata["description"]


def test_published_blood_gas_module_carries_the_status(tmp_path, monkeypatch):
    import os
    import tempfile

    from easyicu.api import extraction

    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for name in ("TMPDIR", "EASYICU_DUCKDB_TEMP_DIR"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
    source = _eicu_panel(tmp_path / "source")
    monkeypatch.setitem(extraction.EXTRACT_MODULES, "blood_gas", ["po2", "ph"])
    monkeypatch.setitem(extraction.EXTRACT_MODULES, "outcome", ["los_icu"])
    result = extraction.extract_database(
        source.config.name, data_path=str(source.base_path), output_dir=str(tmp_path / "out"),
        modules=["blood_gas", "outcome"], native_export_v2=True, verbose=False,
        resource_budget_mb=1024,
    )
    assert not result["modules"]["blood_gas"].get("errors")
    published = pd.read_parquet(tmp_path / "out" / "blood_gas.parquet").sort_values("charttime")
    rows = published.loc[published.po2.notna()]
    assert rows.po2_specimen_status.tolist() == ["label_defined_arterial"] * 2
    assert published.loc[published.ph.notna(), "ph_specimen_status"].tolist() == ["co_measured_arterial"] * 2
    assert pd.api.types.is_string_dtype(published.po2_specimen_status) or published.po2_specimen_status.dtype == object


def test_shared_long_table_merge_keeps_the_status_beside_the_value():
    """Four or more concepts on one long table merge through a partial wide path."""
    from easyicu.table import ICUTable

    wide = pd.DataFrame({"patientunitstayid": [1, 1], "charttime": [0., 1.], "lact": [1.1, 1.3]})

    def table(frame):
        return ICUTable(data=frame, id_columns=["patientunitstayid"], index_column="charttime", value_column="po2")

    plain = pd.DataFrame({"patientunitstayid": [1, 1], "charttime": [0., 1.], "po2": [80., 90.]})
    stamped = plain.assign(po2_specimen_status=["label_defined_arterial", "explicit_arterial"])
    merge = ConceptResolver(load_dictionary())._merge_partial_wide_result
    fast = merge(wide, {"po2": table(plain)}, ["lact", "po2"], {"lact"})
    assert fast.po2.tolist() == [80., 90.]
    # The value-only alignment would drop the companion; the full merge keeps it.
    assert merge(wide, {"po2": table(stamped)}, ["lact", "po2"], {"lact"}) is None
