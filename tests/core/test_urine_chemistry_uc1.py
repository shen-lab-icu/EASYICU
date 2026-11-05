"""UC1-K5/K6: source-specific urine chemistry and pre-existing renal disease."""

import json
from pathlib import Path
import re

import pandas as pd
import pytest

from easyicu.concept import ConceptResolver, _apply_callback
from easyicu.resources import load_dictionary
from tests.support.urine_chemistry_source import sic_precondition_source, urine_chemistry_source

ROOT = Path(__file__).resolve().parents[2]
HISTORY = "sic_precondition_renal_dysfunction"
LABS = {
    "urine_na": (51100, "mmol/L", 400),
    "urine_crea": (51082, "mg/dL", 1000),
    "urine_osm": (51093, "mOsm/kg", 2000),
    "urine_urea": (51104, "mg/dL", 5000),
}


@pytest.mark.parametrize("database", ["miiv", "mimic", "mimic_demo", "eicu", "eicu_demo"])
@pytest.mark.parametrize("bucketed", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_urine_labs_exclude_serum_wrong_units_and_implausible_values(tmp_path, database, bucketed, batch):
    source, expected = urine_chemistry_source(tmp_path, database, bucketed=bucketed)
    resolver = ConceptResolver(load_dictionary())
    actual = {}
    names = list(expected)
    for group in ([names] if batch else [[name] for name in names]):
        actual.update(resolver.load_concepts(
            group, source, merge=False, r_compatible=False, interval=pd.Timedelta(hours=1),
            concept_workers=1, verbose=False,
        ))
    for name, wanted in expected.items():
        table = actual[name]
        frame = table.data.dropna(subset=[name])
        assert dict(zip(frame[table.index_column].astype(float), frame[name])) == wanted


@pytest.mark.parametrize("name", LABS)
def test_urine_lab_sources_have_only_verified_specimens_and_canonical_units(name):
    definition = load_dictionary()[name]
    item, unit, maximum = LABS[name]
    assert definition.units == [unit]
    assert definition.minimum == 0 and definition.maximum == maximum
    for database in ("miiv", "mimic", "mimic_demo"):
        source, = definition.sources[database]
        assert source.table == "labevents" and source.sub_var == "itemid" and source.ids == [item]
    # No checked-in AUMC/HiRID/SICdb dictionary proves a urine-specific item.
    assert not {"aumc", "hirid", "sic"} & definition.sources.keys()
    if name in {"urine_na", "urine_crea"}:
        label = {"urine_na": "urinary sodium", "urine_crea": "urinary creatinine"}[name]
        for database in ("eicu", "eicu_demo"):
            source, = definition.sources[database]
            assert source.table == "lab" and source.sub_var == "labname" and source.ids == [label]
    else:
        assert not {"eicu", "eicu_demo"} & definition.sources.keys()


@pytest.mark.parametrize("database", ["miiv", "mimic", "mimic_demo", "eicu", "eicu_demo"])
def test_urine_sodium_equivalent_milliequivalents_are_explicitly_relabelled(database):
    source, = load_dictionary()["urine_na"].sources[database]
    column = "labmeasurenameinterface" if database.startswith("eicu") else "valueuom"
    frame = pd.DataFrame({"urine_na": [40.], column: ["mEq/L"]})
    out = _apply_callback(frame, source, "urine_na", unit_column=column)
    assert out.urine_na.tolist() == [40.]
    assert out[column].tolist() == ["mmol/L"]


def test_sic_renal_precondition_uses_reference_labels_and_preserves_unknowns(tmp_path):
    source = sic_precondition_source(tmp_path)
    table = ConceptResolver(load_dictionary()).load_concepts(
        [HISTORY], source, merge=False, r_compatible=False, concept_workers=1, verbose=False,
    )[HISTORY]
    values = table.data.set_index("CaseID")[HISTORY]
    assert values.dropna().to_dict() == {1: True, 2: False}
    assert not values.reindex([3, 4, 5, 6, 7]).notna().any()
    assert table.index_column is None


def test_sic_renal_precondition_missing_reference_table_is_unavailable(tmp_path):
    from easyicu.concept.errors import ConceptExtractionUnavailable

    source = sic_precondition_source(tmp_path)
    (tmp_path / "d_references.parquet").unlink()
    definition = load_dictionary()[HISTORY]
    with pytest.raises(ConceptExtractionUnavailable):
        _apply_callback(pd.DataFrame({"CaseID": [1], "FieldID": [3104], HISTORY: [90001]}),
                        definition.sources["sic"][0], HISTORY, data_source=source)


def test_sic_precondition_without_fieldid_is_unavailable(tmp_path):
    from easyicu.concept.errors import ConceptExtractionUnavailable

    source = sic_precondition_source(tmp_path)
    definition = load_dictionary()[HISTORY]
    # Converted tables predating the FieldID schema must not decode arbitrary
    # Yes/No references as renal history.
    with pytest.raises(ConceptExtractionUnavailable, match="FieldID"):
        _apply_callback(pd.DataFrame({"CaseID": [1], HISTORY: [90001]}),
                        definition.sources["sic"][0], HISTORY, data_source=source)


def test_sic_renal_precondition_obeys_the_requested_case_subset(tmp_path):
    source = sic_precondition_source(tmp_path)
    table = ConceptResolver(load_dictionary()).load_concepts(
        [HISTORY], source, merge=False, patient_ids={"CaseID": [2]},
        r_compatible=False, concept_workers=1, verbose=False,
    )[HISTORY]
    assert table.data[["CaseID", HISTORY]].to_dict("records") == [{"CaseID": 2, HISTORY: False}]


def test_sic_renal_precondition_empty_source_needs_no_reference_lookup():
    source, = load_dictionary()[HISTORY].sources["sic"]
    out = _apply_callback(pd.DataFrame({"CaseID": pd.Series(dtype=int), HISTORY: pd.Series(dtype=int)}),
                          source, HISTORY, data_source=None)
    assert out.empty and str(out[HISTORY].dtype) == "boolean"


def test_generated_sic_source_preserves_the_precondition_field_and_labels(tmp_path):
    from easyicu.config import load_src_cfg
    from easyicu.datasource import ICUDataSource
    from easyicu.synthetic_sources import generate_source

    root = tmp_path / "synthetic"
    generate_source(root, "sic", stays=3)
    assert pd.read_parquet(root / "data_ref.parquet").FieldID.eq(3104).all()
    source = ICUDataSource(load_src_cfg("sic"), base_path=root, enable_cache=False)
    table = ConceptResolver(load_dictionary()).load_concepts(
        [HISTORY], source, merge=False, r_compatible=False, concept_workers=1, verbose=False,
    )[HISTORY]
    assert table.data.set_index("CaseID")[HISTORY].to_dict() == {1: True, 2: False, 3: True}


@pytest.mark.parametrize("streamed", [False, True])
def test_sic_renal_precondition_survives_native_module_publication(tmp_path, monkeypatch, streamed):
    from easyicu.api.extraction import extract_database
    from easyicu.synthetic_sources import generate_source

    monkeypatch.setenv("EASYICU_NATIVE_V2_REAL_ROOT", "")
    root, output = tmp_path / "source", tmp_path / "export"
    generate_source(root, "sic", stays=3)
    result = extract_database(
        "sic", data_path=str(root), output_dir=str(output), modules=["renal_urine_chem", "outcome"],
        stream_output_batches=streamed, batch_size=1 if streamed else None,
        native_export_v2=True, resource_budget_mb=2048, verbose=False,
    )
    assert not result["modules"]["renal_urine_chem"].get("errors")
    values = pd.read_parquet(output / "renal_urine_chem.parquet")[["stay_id", HISTORY]].drop_duplicates()
    assert values.set_index("stay_id")[HISTORY].to_dict() == {1: True, 2: False, 3: True}


def test_urine_concept_allowlist_retains_every_key_with_compact_rendering(monkeypatch):
    from easyicu.research_agent.agents import planner

    names = [*LABS, HISTORY]
    monkeypatch.setattr(planner, "known_concept_ids", lambda **kwargs: names)
    rendered = planner._format_concept_id_allowlist([])
    entries = [line for line in rendered.splitlines() if line.startswith("- `")]
    assert len(entries) == 1
    assert re.findall(r"`([^`]+)`", entries[0]) == sorted(names)


@pytest.mark.parametrize("name", [*LABS, HISTORY])
def test_urine_and_history_concepts_are_registered_once_in_their_own_module(name):
    from easyicu.api.extraction import EXTRACT_MODULES
    from easyicu.concept.catalog import CONCEPT_DB_COVERAGE, CONCEPT_DESCRIPTIONS, CONCEPT_DICTIONARY

    assert EXTRACT_MODULES["renal_urine_chem"].count(name) == 1
    assert name not in EXTRACT_MODULES["renal"]
    assert name in CONCEPT_DICTIONARY and name in CONCEPT_DESCRIPTIONS
    assert CONCEPT_DB_COVERAGE[name] == (1 if name == HISTORY else 3 if name in {"urine_na", "urine_crea"} else 2)
    payload = json.loads((ROOT / "src/easyicu/data/concept-dict.json").read_text())
    assert payload[name]["category"] == "renal"
    assert payload[name]["_comment"]
    if name == HISTORY:
        assert payload[name]["target"] == "id_tbl"
        assert set(payload[name]["sources"]) == {"sic"}
        assert "ICD" in payload[name]["_comment"]
