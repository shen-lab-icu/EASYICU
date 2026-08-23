"""Truthfulness gates for the first high-frequency method-adapter batch."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from easyicu.research_agent.planning.analysis_types import list_analysis_types
from easyicu.research_agent.planning.method_adapter_catalog import (
    HIGH_FREQUENCY_METHOD_ADAPTERS,
    MethodAdapterGapError,
    method_adapter_catalog_receipt,
    require_method_adapter_contract,
)
from easyicu.research_agent.planning.scientific_action_catalog import (
    planner_scientific_action_guide,
    scientific_actions_for_analysis_type,
)


def _all_actions():
    actions = {}
    for analysis_type in list_analysis_types():
        for action in scientific_actions_for_analysis_type(analysis_type.key).actions:
            actions.setdefault(action.action_id, action)
    return actions


def test_first_batch_has_19_unique_case_neutral_analysis_only_adapters() -> None:
    assert len(HIGH_FREQUENCY_METHOD_ADAPTERS) == 19
    assert len({item.adapter_id for item in HIGH_FREQUENCY_METHOD_ADAPTERS}) == 19
    assert len({item.action_id for item in HIGH_FREQUENCY_METHOD_ADAPTERS}) == 19
    assert {item.claim_ceiling for item in HIGH_FREQUENCY_METHOD_ADAPTERS} == {
        "analysis_only"
    }
    rendered = repr(HIGH_FREQUENCY_METHOD_ADAPTERS).casefold()
    assert not re.search(r"\b(e1|sepsis|h3)\b", rendered)
    typed_subcontracts = {
        item.action_id
        for item in HIGH_FREQUENCY_METHOD_ADAPTERS
        if item.scope == "typed_subcontract"
    }
    assert typed_subcontracts == {
        "association.ordinal_trend",
        "time_to_event.km_logrank",
        "time_to_event.ph_check",
        "phenotyping.cluster_sizes",
        "phenotyping.outcome_by_cluster",
    }


def test_every_adapter_names_a_real_owner_and_real_validation_tests() -> None:
    repo = Path(__file__).resolve().parents[2]
    for item in HIGH_FREQUENCY_METHOD_ADAPTERS:
        module = importlib.import_module(item.owner_module)
        assert callable(getattr(module, item.owner_entrypoint, None)), item.adapter_id
        for reference in item.validation_test_refs:
            relative, separator, test_name = reference.partition("::")
            assert separator and test_name.startswith("test_")
            source_path = repo / relative
            assert source_path.is_file(), reference
            source = source_path.read_text(encoding="utf-8")
            assert f"def {test_name}(" in source, reference


def test_adapter_contracts_project_into_the_planner_action_surface() -> None:
    actions = _all_actions()
    assert set(item.action_id for item in HIGH_FREQUENCY_METHOD_ADAPTERS) <= set(
        actions
    )
    for item in HIGH_FREQUENCY_METHOD_ADAPTERS:
        action = actions[item.action_id]
        assert action.method_adapter == item
        if item.scope == "full_action":
            assert action.execution_mode == "host_owned"

    prediction_guide = planner_scientific_action_guide("prediction_model")
    assert "prediction_decision_curve_v1[full_action]" in prediction_guide
    names_only = planner_scientific_action_guide(
        "prediction_model", detail="names_only"
    )
    assert "typed_adapters:" in names_only
    assert "prediction.decision_curve=prediction_decision_curve_v1" in names_only

    actions = _all_actions()
    support_only = {
        action.action_id
        for action in actions.values()
        if action.adapter_status == "supporting_only"
    }
    assert support_only == {
        "association.multiple_testing",
        "association.evalue",
        "prediction.subgroup_fairness",
        "causal_emulation.covariate_balance",
        "causal_emulation.evalue",
    }
    causal_guide = planner_scientific_action_guide(
        "causal_inference", detail="names_only"
    )
    assert "host_support_only:" in causal_guide
    assert "causal_emulation.covariate_balance" in causal_guide


def test_unregistered_adapter_fails_closed_with_a_stable_code() -> None:
    with pytest.raises(MethodAdapterGapError) as captured:
        require_method_adapter_contract("prediction.reclassification")
    assert captured.value.code == "method_adapter_not_registered"
    assert captured.value.action_id == "prediction.reclassification"


def test_adapter_catalog_receipt_is_stable_and_complete() -> None:
    first = method_adapter_catalog_receipt()
    second = method_adapter_catalog_receipt()
    assert first == second
    assert first["schema_version"] == "easyicu.method_adapter_catalog/1"
    assert first["adapter_count"] == 19
    assert first["claim_ceiling"] == "analysis_only"
    assert re.fullmatch(r"[0-9a-f]{64}", str(first["catalog_sha256"]))
    assert first["action_ids"] == [
        item.action_id for item in HIGH_FREQUENCY_METHOD_ADAPTERS
    ]
