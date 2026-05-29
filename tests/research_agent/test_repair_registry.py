from __future__ import annotations

import json
import re
from pathlib import Path

from easyicu.research_agent import code_repair
from easyicu.research_agent.repair_registry import (
    RepairClass,
    RepairLedger,
    assert_registry_invariants,
    make_repair_provenance,
    repair_metadata_for,
)


def test_repair_registry_invariants_hold() -> None:
    assert_registry_invariants()


def test_literal_code_repair_ids_are_classified() -> None:
    source = Path(code_repair.__file__).read_text(encoding="utf-8")
    repair_ids = set(re.findall(r"repair_name\s*=\s*['\"]([^'\"]+)['\"]", source))
    assert repair_ids
    unclassified = [
        repair_id
        for repair_id in sorted(repair_ids)
        if repair_metadata_for(repair_id).classification_source.startswith("fallback:")
    ]
    assert unclassified == []


def test_dynamic_repair_id_patterns_are_classified() -> None:
    assert (
        repair_metadata_for("generic_v15_table_one_fallback_v1").repair_class
        is RepairClass.METHOD_SUBSTITUTION
    )
    assert (
        repair_metadata_for("strip_fake_easyicu_import_easyicu_foo_v1").repair_class
        is RepairClass.SYNTACTIC
    )
    assert (
        repair_metadata_for("undefined_helper_stub_to_json_serializable_v1").repair_class
        is RepairClass.STRUCTURAL
    )


def test_nonconvergence_fallback_is_method_substitution() -> None:
    metadata = repair_metadata_for("validation_nonconvergence_fallback_v1")
    assert metadata.repair_class is RepairClass.METHOD_SUBSTITUTION
    assert metadata.introduces_numbers is True
    assert metadata.requires_disclosure is True


def test_contract_fill_requires_selection_rule() -> None:
    metadata = repair_metadata_for("categorical_primary_association_selection_v1")
    assert metadata.repair_class is RepairClass.CONTRACT_FILL
    assert metadata.selection_rule_required is True


def test_repair_ledger_writes_provenance_json(tmp_path: Path) -> None:
    ledger = RepairLedger(tmp_path / "repairs_applied.json")
    provenance = ledger.append_application(
        repair_id="statsmodels_endog_exog_index_align_v1",
        step_id="04_model",
        trigger={"error_type": "ValueError"},
        transformation="Aligned endog/exog indices.",
        before_text="before",
        after_text="after",
    )
    assert provenance.repair_class == RepairClass.STRUCTURAL.value

    payload = json.loads((tmp_path / "repairs_applied.json").read_text())
    assert payload["schema_version"] == "easyicu.repair_ledger/1"
    assert payload["repairs"][0]["repair_id"] == "statsmodels_endog_exog_index_align_v1"
    assert payload["repairs"][0]["before_hash"].startswith("sha256:")
    assert payload["repairs"][0]["after_hash"].startswith("sha256:")


def test_make_repair_provenance_conservatively_classifies_unknown() -> None:
    provenance = make_repair_provenance(
        repair_id="future_unreviewed_repair_v1",
        step_id="01_step",
    )
    assert provenance.repair_class == RepairClass.METHOD_SUBSTITUTION.value
    assert provenance.classification_source == "fallback:unknown_method_substitution"
    assert provenance.requires_disclosure is True
