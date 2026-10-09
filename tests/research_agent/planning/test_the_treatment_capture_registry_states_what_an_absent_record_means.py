"""The treatment capture registry states what an absent record means.

A target trial reads its deferring strategy from absent records, so whether
an absence means "not given" is stated once per database and treatment
concept, beside the concept dictionaries, never inferred from a question.
Each drug an entry lists is tied to the dictionary component or source item
it is read from, and those ties are checked against the dictionaries; the
absence vocabulary is the endpoint owner's, every entry carries who stated
it, and the registry is closed: an entry whose drugs its definition does not
read, a drug no class names, or a concept stated twice is refused.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from easyicu.concept.export_metadata import concept_declares_event_status
from easyicu.research_agent.contracts.endpoint import EndpointAbsenceSemantics
from easyicu.research_agent.planning.treatment_capture import (
    TREATMENT_CAPTURE_REGISTRY_SCHEMA_VERSION,
    AbsentInCapture,
    TreatmentCaptureRegistry,
    load_treatment_capture_registry,
    packaged_treatment_capture_registry,
    treatment_capture_registry_path,
)

_DATA = Path(treatment_capture_registry_path()).parent


def _raw() -> dict[str, Any]:
    return json.loads(treatment_capture_registry_path().read_text(encoding="utf-8"))


def _dictionary(name: str) -> dict[str, Any]:
    return json.loads((_DATA / name).read_text(encoding="utf-8"))


def test_the_packaged_registry_loads_with_the_digest_of_its_bytes() -> None:
    loaded = packaged_treatment_capture_registry()

    assert loaded.registry.schema_version == TREATMENT_CAPTURE_REGISTRY_SCHEMA_VERSION
    assert (
        loaded.sha256
        == hashlib.sha256(treatment_capture_registry_path().read_bytes()).hexdigest()
    )
    assert load_treatment_capture_registry().registry == loaded.registry


def _source_note_labels(rows: list[dict[str, Any]]) -> dict[int, str]:
    """The item labels a dictionary's source note gives, as ``id=Label (rows)``."""

    return {
        int(item_id): label
        for row in rows
        for item_id, label in re.findall(
            r"(\d+)=(.+?) \(\d+\)(?:, |$)", str(row.get("_comment", ""))
        )
    }


def test_each_drug_is_tied_to_the_dictionary_definition_it_is_read_from() -> None:
    registry = packaged_treatment_capture_registry().registry
    dictionaries: dict[str, dict[str, Any]] = {}
    for entry in registry.entries:
        definition = entry.definition
        source = dictionaries.setdefault(
            definition.dictionary, _dictionary(definition.dictionary)
        )
        concept = source[entry.concept]
        if definition.components:
            assert {item.component for item in definition.components} == set(
                concept["concepts"]
            ), entry.concept
            for item in definition.components:
                # A component's own definition names its drug first.
                description = str(source[item.component]["description"]).lower()
                assert description.startswith(item.agent.replace("_", " ")), (
                    item.component
                )
        if definition.source_items:
            rows = concept["sources"][entry.database]
            items = {item_id for row in rows for item_id in row.get("ids", ())}
            assert {item.item_id for item in definition.source_items} == items, (
                entry.concept
            )
            labels = _source_note_labels(rows)
            for item in definition.source_items:
                assert labels[item.item_id] == item.label, item.item_id
                # The item's label names its drug first.
                assert item.label.lower().replace(" ", "_").startswith(item.agent), (
                    item.item_id
                )
        assert set(entry.agents) == definition.agents, entry.concept


def test_each_entry_is_a_concept_that_records_an_event_status() -> None:
    for entry in packaged_treatment_capture_registry().registry.entries:
        assert concept_declares_event_status(entry.concept), entry.concept


def test_the_absence_vocabulary_is_the_endpoint_owners() -> None:
    assert set(get_args(AbsentInCapture)) - {"unknown"} <= set(
        get_args(EndpointAbsenceSemantics)
    )


def test_each_entry_says_who_stated_it_and_what_it_assumes() -> None:
    for entry in packaged_treatment_capture_registry().registry.entries:
        assert entry.basis == "development_assumption"
        assert entry.declared_by and re.fullmatch(
            r"\d{4}-\d{2}-\d{2}", entry.declared_at
        )
        # An infusion started before ICU admission is not in ICU charting.
        assert entry.pre_admission_visible is False
        assert entry.capture_setting == "icu"


def test_the_registry_names_no_development_question() -> None:
    text = treatment_capture_registry_path().read_text(encoding="utf-8")

    assert not re.search(r"\b[EMH][1-3]\b", text)
    assert not re.search(r"canonical[-_ ]?9|\bdev9\b", text, re.IGNORECASE)


def _refused(change, match: str) -> None:
    data = copy.deepcopy(_raw())
    change(data)
    with pytest.raises(ValidationError, match=match):
        TreatmentCaptureRegistry.model_validate(data)


def test_an_entry_naming_a_drug_no_class_names_is_refused() -> None:
    def change(data: dict[str, Any]) -> None:
        entry = data["entries"][0]
        entry["agents"].append("levosimendan")
        entry["definition"]["components"].append(
            {"component": "levo_dur", "agent": "levosimendan"}
        )

    _refused(change, "records agents no class names")


def test_an_entry_whose_drugs_its_definition_does_not_read_is_refused() -> None:
    # Listed without a definition that reads it ...
    _refused(
        lambda data: data["entries"][0]["agents"].append("milrinone"),
        "lists agents its definition does not read",
    )

    # ... or read by the definition without being listed.
    def unlisted(data: dict[str, Any]) -> None:
        data["entries"][0]["definition"]["components"].append(
            {"component": "milrinone_dur", "agent": "milrinone"}
        )

    _refused(unlisted, "lists agents its definition does not read")


def test_a_definition_tying_one_component_twice_is_refused() -> None:
    def change(data: dict[str, Any]) -> None:
        components = data["entries"][0]["definition"]["components"]
        components.append(dict(components[0]))

    _refused(change, "names each component and source item once")


def test_a_concept_stated_twice_for_one_database_is_refused() -> None:
    _refused(
        lambda data: data["entries"].append(copy.deepcopy(data["entries"][0])),
        "state each database's concept once",
    )


def test_a_class_name_that_is_no_token_is_refused() -> None:
    _refused(
        lambda data: data["treatment_classes"].update(
            {"Vaso Active": {"agents": ["dopamine"], "note": "Not a token."}}
        ),
        "is not a token",
    )


def test_a_definition_naming_nothing_is_refused() -> None:
    def change(data: dict[str, Any]) -> None:
        data["entries"][0]["definition"] = {"dictionary": "concept-dict.json"}

    _refused(change, "names its components or its source items")


def test_an_unknown_field_or_basis_is_refused() -> None:
    _refused(
        lambda data: data["entries"][0].update({"verified": True}),
        "Extra inputs are not permitted",
    )
    _refused(
        lambda data: data["entries"][0].update({"basis": "model_inferred"}),
        "development_assumption",
    )


def test_a_lookup_reads_the_database_as_the_registry_states_it() -> None:
    registry = packaged_treatment_capture_registry().registry

    assert registry.entry(" MIIV ", "vaso_ind") is not None
    assert registry.entry("eicu", "vaso_ind") is None
    assert registry.class_agents("no_such_class") is None
    assert "vasopressin" in registry.class_agents("vasopressor")
