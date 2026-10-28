"""Provenance is extraction metadata, independent of any clinical case."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from easyicu.concept.schema import ConceptDictionary, ConceptDefinition
from tests.support.data_provenance import synthetic_dictionary


def test_source_conversion_codes_and_unknown_are_recorded_without_guessing():
    from easyicu.concept.provenance import capture_data_provenance

    snapshot = capture_data_provenance(
        synthetic_dictionary(), concepts=["lab_signal", "coded_condition", "unknown"],
        source_database="MIIV",
    )
    lab, diagnosis, unknown = snapshot.concepts
    assert lab.source_database == "miiv"
    assert lab.sources[0].table == "synthetic_labs"
    assert lab.sources[0].item_ids == (71, 72)
    assert lab.canonical_unit == "mmol/L"
    assert lab.sources[0].unit_variable == "unit"
    assert lab.sources[0].callback.startswith("convert_unit(")
    conversion = lab.sources[0].unit_conversions[0]
    assert conversion.source_unit_pattern == "mg/dL"
    assert conversion.target_unit == "mmol/L"
    assert conversion.factor == 0.1
    assert conversion.offset == 0.0
    assert conversion.application == "source_callback_rule"
    from easyicu.concept.callback_apply import _apply_callback

    source = synthetic_dictionary()["lab_signal"].sources["miiv"][0]
    converted = _apply_callback(
        pd.DataFrame({"lab_signal": [20.0], "unit": ["mg/dL"]}),
        source, "lab_signal", "unit",
    )
    assert converted["lab_signal"].iloc[0] == 20 * conversion.factor
    assert diagnosis.definition == "Synthetic condition defined by the declared code list"
    assert diagnosis.definition_version == "v1"
    assert diagnosis.sources[0].codes[0].description == "Synthetic criterion one"
    assert diagnosis.sources[0].codes[1].description is None
    assert diagnosis.sources[0].codes[1].description_status == "not_recorded"
    assert unknown.status == "not_recorded"
    assert unknown.sources == ()
    assert unknown.definition is None


def test_snapshot_pins_actual_dictionary_and_resolver_inputs():
    from easyicu.concept.provenance import capture_data_provenance, DataProvenanceSnapshot

    dictionary = synthetic_dictionary()
    first = capture_data_provenance(dictionary, concepts=["lab_signal"], source_database="miiv")
    dictionary.update(ConceptDictionary({
        "lab_signal": replace(dictionary["lab_signal"], description="Revised later"),
    }))
    second = capture_data_provenance(dictionary, concepts=["lab_signal"], source_database="miiv")
    assert first.dictionary_sha256 != second.dictionary_sha256
    assert first.snapshot_sha256 != second.snapshot_sha256
    assert first.concepts[0].definition == "Synthetic laboratory measurement"
    changed = first.model_dump(mode="json")
    changed["concepts"][0]["sources"][0]["item_ids"] = [999]
    with pytest.raises(ValueError, match="digest"):
        DataProvenanceSnapshot.model_validate(changed)
    inherited = capture_data_provenance(
        dictionary, concepts=["lab_signal"], source_database="miiv_demo",
        class_prefixes=["miiv"],
    )
    assert inherited.resolver_inputs_sha256 != second.resolver_inputs_sha256
    assert inherited.concepts[0].dictionary_source_database == "miiv"


def test_dependencies_are_frozen_and_arbitrary_source_parameters_are_not_published():
    from easyicu.concept.provenance import capture_data_provenance

    dictionary = synthetic_dictionary()
    dictionary.update(ConceptDictionary({"derived_signal": ConceptDefinition(
        name="derived_signal", sources={}, sub_concepts=["lab_signal"], callback="synthetic_derive",
    )}))
    dictionary["lab_signal"].sources["miiv"][0].params["patient_rows"] = [{"value": 1}]
    snapshot = capture_data_provenance(dictionary, concepts=["derived_signal"], source_database="miiv")
    assert [c.concept_id for c in snapshot.concepts] == ["derived_signal", "lab_signal"]
    assert snapshot.concepts[0].derivation_inputs == ("lab_signal",)
    assert snapshot.concepts[0].derivation_callback == "synthetic_derive"
    assert "patient_rows" not in snapshot.model_dump_json()


@pytest.mark.parametrize(("rule", "factor", "offset"), [
    ("binary_op(`*`, 3)", 3.0, 0.0),
    ("binary_op(`+`, 4)", 1.0, 4.0),
    ("binary_op(`-`, 2)", 1.0, -2.0),
    ("fahr_to_cels", 5 / 9, -32 * 5 / 9),
    ("binary_op(`^`, 2)", None, None),
])
def test_conversion_rules_keep_affine_factors_or_explicitly_unknown(rule, factor, offset):
    from easyicu.concept.provenance import capture_data_provenance

    dictionary = synthetic_dictionary()
    dictionary["lab_signal"].sources["miiv"][0].callback = (
        f"combine_callbacks(convert_unit({rule}, 'target', 'source'), identity)"
    )
    conversion = capture_data_provenance(
        dictionary, concepts=["lab_signal"], source_database="miiv",
    ).concepts[0].sources[0].unit_conversions[0]
    assert conversion.factor == factor
    assert conversion.offset == offset
    assert conversion.blank_unit_matches is (rule == "fahr_to_cels")


def test_forwarded_dependencies_and_specific_source_override_are_recorded():
    from easyicu.concept.provenance import capture_data_provenance
    from easyicu.concept.schema import ConceptSource

    dictionary = synthetic_dictionary()
    dictionary["lab_signal"].sources["miiv_demo"] = [ConceptSource(table="specific", ids=[91])]
    dictionary.update(ConceptDictionary({"forwarded": ConceptDefinition(
        name="forwarded", sources={}, callback="fwd_concept('lab_signal')",
    )}))
    snapshot = capture_data_provenance(
        dictionary, concepts=["forwarded"], source_database="miiv_demo", class_prefixes=["miiv"],
    )
    assert snapshot.concepts[0].derivation_inputs == ("lab_signal",)
    assert snapshot.concepts[1].sources[0].table == "specific"
    assert snapshot.concepts[1].sources[0].item_ids == (91,)
    assert len(snapshot.concepts[1].sources) == 1


def test_parameter_and_resource_identity_change_the_resolver_binding():
    from easyicu.concept.provenance import capture_data_provenance

    kwargs = dict(concepts=["lab_signal"], source_database="miiv")
    first = capture_data_provenance(synthetic_dictionary(), **kwargs)
    configured = capture_data_provenance(
        synthetic_dictionary(), **kwargs, resolver_parameters={"window_hours": 12},
        resource_inputs_sha256="a" * 64,
    )
    assert configured.resolver_inputs_sha256 != first.resolver_inputs_sha256
    assert configured.resource_inputs_sha256 == "a" * 64
    assert configured.resolver_parameters[0].values == (12,)


def test_closed_types_refuse_invented_unknowns_and_inconsistent_descriptions():
    from easyicu.concept.provenance import CodeProvenance, ConceptProvenance

    with pytest.raises(ValueError, match="invented"):
        ConceptProvenance(concept_id="unknown", status="not_recorded", source_database="miiv", definition="Guess")
    with pytest.raises(ValueError, match="status"):
        CodeProvenance(code="Z01", description="Guess")


def test_capture_refuses_dictionary_or_implementation_drift(monkeypatch):
    from easyicu.config import DataSourceConfig
    from easyicu.concept.provenance_capture import ExtractionProvenanceCapture

    capture = ExtractionProvenanceCapture()
    capture.record(
        dictionary=synthetic_dictionary(), source_config=DataSourceConfig(name="miiv"),
        concepts=["lab_signal"],
        cohort_contract={"icd_include": ["Z01"], "age_min": 18},
    )
    snapshot = capture.finish()["snapshots"][0]
    assert snapshot["cohort_filters"][0]["codes"][0]["description_status"] == "not_recorded"
    assert snapshot["cohort_filters"][1]["values"] == [18]
    monkeypatch.setattr(capture, "_input_identity", lambda: "changed")
    with pytest.raises(ValueError, match="inputs changed"):
        capture.finish()


def test_native_api_records_the_selected_cached_resolver_before_any_read(tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace
    from easyicu.config import DataSourceConfig
    from easyicu.concept.provenance_capture import ExtractionProvenanceCapture

    owner = importlib.import_module("easyicu.api.concepts")
    dictionary = synthetic_dictionary()
    config = DataSourceConfig(name="miiv", class_prefix=["miiv"])
    loader = SimpleNamespace(
        concept_resolver=SimpleNamespace(dictionary=dictionary),
        datasource=SimpleNamespace(config=config),
    )
    monkeypatch.setattr(owner, "_get_global_loader", lambda **_: loader)
    capture = ExtractionProvenanceCapture()

    class BeforeDataRead(RuntimeError):
        pass

    record = capture.recorder(concepts=["lab_signal"], cohort_contract={})

    def stop_after_record(actual_dictionary, actual_config):
        record(actual_dictionary, actual_config)
        raise BeforeDataRead

    with pytest.raises(BeforeDataRead):
        owner.load_concepts(
            ["lab_signal"], database="miiv", data_path=tmp_path,
            patient_ids={"stay_id": [1]}, verbose=False,
            _provenance_sink=stop_after_record,
        )
    snapshot = capture.finish()["snapshots"][0]
    assert snapshot["concepts"][0]["sources"][0]["table"] == "synthetic_labs"
    assert snapshot["source_config_sha256"] is not None
    dictionary["lab_signal"].sources["miiv"][0].ids.append(999)
    with pytest.raises(ValueError, match="resolver inputs changed"):
        capture.finish()


def test_native_observer_is_removed_after_failure():
    from easyicu.concept.provenance import observing_resolver_inputs, record_resolver_inputs

    def unused_sink(dictionary, config):
        pytest.fail("observer escaped its extraction")

    with pytest.raises(RuntimeError), observing_resolver_inputs(unused_sink):
        raise RuntimeError("synthetic interrupted extraction")
    record_resolver_inputs({}, object())


@pytest.mark.parametrize("violation", ["database", "duplicate", "mixed_export"])
def test_snapshot_rejects_inconsistent_concept_or_export_identity(violation):
    from easyicu.concept.provenance import (
        capture_data_provenance, DataProvenanceSnapshot, ExportDataProvenance, metadata_digest,
    )

    first = capture_data_provenance(synthetic_dictionary(), concepts=["lab_signal"], source_database="miiv")
    if violation == "mixed_export":
        second = capture_data_provenance(synthetic_dictionary(), concepts=["unknown"], source_database="eicu")
        with pytest.raises(ValueError, match="database"):
            ExportDataProvenance(snapshots=(first, second))
        return
    payload = first.model_dump(mode="json", exclude={"snapshot_sha256"})
    if violation == "database":
        payload["concepts"][0]["source_database"] = "eicu"
    else:
        payload["concepts"].append(payload["concepts"][0])
    with pytest.raises(ValueError, match="database|duplicate"):
        DataProvenanceSnapshot(**payload, snapshot_sha256=metadata_digest(payload))
