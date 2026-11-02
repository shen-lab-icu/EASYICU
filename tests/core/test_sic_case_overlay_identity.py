"""The SIC extension remains distinct from the historically sealed base."""
from __future__ import annotations

import hashlib
import json
from importlib import resources

import pytest

from easyicu.resources import SIC_CASE_OVERLAY_CONCEPTS, SIC_CASE_OVERLAY_FILE, SIC_CASE_OVERLAY_SHA256, load_dictionary
from easyicu.research_agent.concept_dict_audit import ConceptDictDriftError, assert_dict_matches, compute_concept_dict_fingerprint
from easyicu.research_agent.orchestration.profiles import E1_PROGRESSIVE_PLANNER_CANARY_2026_10_26, E1_PROGRESSIVE_PLANNER_CANARY_SIC_OVERLAY


def test_only_sic_source_selects_the_exact_overlay():
    base = load_dictionary()
    sic = load_dictionary(source_name="sic")
    eicu = load_dictionary(source_name="eicu")
    for name in SIC_CASE_OVERLAY_CONCEPTS:
        assert base.get(name) is None
        assert eicu.get(name) is None
        assert sic.get(name) is not None
    data = resources.files("easyicu.data").joinpath(SIC_CASE_OVERLAY_FILE).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SIC_CASE_OVERLAY_SHA256
    assert set(json.loads(data)) == SIC_CASE_OVERLAY_CONCEPTS


def test_planner_database_catalog_exposes_overlay_only_for_sic():
    from easyicu.research_agent.acquisition.catalog import build_database_capability_catalog

    sic = {item.concept_id for item in build_database_capability_catalog("sic").concepts}
    eicu = {item.concept_id for item in build_database_capability_catalog("eicu").concepts}
    miiv = {item.concept_id for item in build_database_capability_catalog("miiv").concepts}
    assert SIC_CASE_OVERLAY_CONCEPTS <= sic
    assert not SIC_CASE_OVERLAY_CONCEPTS.intersection(eicu)
    assert not SIC_CASE_OVERLAY_CONCEPTS.intersection(miiv)


def test_tampered_overlay_file_is_rejected(tmp_path, monkeypatch):
    import easyicu.resources as resource_owner

    (tmp_path / SIC_CASE_OVERLAY_FILE).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(resource_owner.resources, "files", lambda _package: tmp_path)
    with pytest.raises(ValueError, match="overlay_identity_mismatch"):
        resource_owner._sic_case_overlay()


def test_custom_changed_definition_cannot_claim_packaged_overlay_identity():
    from easyicu.concept.provenance import capture_data_provenance

    dictionary = load_dictionary(source_name="sic")
    dictionary.get("death_during_case").description = "altered meaning"
    with pytest.raises(ValueError, match="overlay_definition_mismatch"):
        capture_data_provenance(dictionary, source_database="sic", concepts=["death_during_case"])


def test_custom_extra_cannot_replace_a_pinned_sic_definition(tmp_path):
    (tmp_path / "changed.json").write_text(
        json.dumps({"death_during_case": {"description": "changed meaning"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="overlay_definition_mismatch"):
        load_dictionary(source_name="sic", directories=[tmp_path], extras=["changed"])


def test_native_provenance_names_overlay_only_for_sic():
    from easyicu.api.extraction import _native_export_runtime_provenance

    sic = _native_export_runtime_provenance(source_name="sic")
    eicu = _native_export_runtime_provenance(source_name="eicu")
    assert sic["concept_dictionary_sha256"] == eicu["concept_dictionary_sha256"]
    assert sic["sic_case_overlay_sha256"] == SIC_CASE_OVERLAY_SHA256
    assert "sic_case_overlay_sha256" not in eicu
    assert sic["concept_catalog_sha256"] != eicu["concept_catalog_sha256"]


def test_historical_profile_does_not_authorize_installed_overlay():
    fingerprint = compute_concept_dict_fingerprint()
    old = E1_PROGRESSIVE_PLANNER_CANARY_2026_10_26
    with pytest.raises(ConceptDictDriftError, match="overlay SHA mismatch"):
        assert_dict_matches(fingerprint, expected_concept_dict_sha=old.expected_concept_dict_sha,
                            expected_sofa2_dict_sha=old.expected_sofa2_dict_sha)
    current = E1_PROGRESSIVE_PLANNER_CANARY_SIC_OVERLAY
    assert assert_dict_matches(fingerprint, expected_concept_dict_sha=current.expected_concept_dict_sha,
                               expected_sofa2_dict_sha=current.expected_sofa2_dict_sha,
                               expected_sic_case_overlay_sha=current.expected_sic_case_overlay_sha) == []


@pytest.mark.parametrize("budget_mode", ["metadata_only", "full_reviewed"])
@pytest.mark.parametrize("runner_kind", ["host", "docker"])
def test_actual_web_profile_preflight_accepts_pinned_overlay(monkeypatch, budget_mode, runner_kind):
    from easyicu.webserver import capabilities
    from easyicu.webserver.research_launch_runtime import _require_profile_dictionaries

    monkeypatch.setattr(capabilities, "pubmed_connector_block_reason", lambda **_: None)
    _require_profile_dictionaries(budget_mode=budget_mode, runner_kind=runner_kind)
