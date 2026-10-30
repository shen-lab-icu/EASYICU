"""No patient rows: approved-candidate source-axis attestation at the host boundary."""

import copy
import json
from types import SimpleNamespace

import pytest

from easyicu.databases.research_capabilities import research_capability_profile
from easyicu.databases.time_origin import SIC_CARE_UNIT_ORIGIN
from easyicu.research_agent.canonical_json import canonical_sha256
from easyicu.webserver import agent_pipeline_runs as owner
from easyicu.webserver import time_origin_source_authority as authority


def _candidate():
    identity = {"database": "sic", "registration_id": "src_example", "version": "1.0.6"}
    selection = {"requested_origin": "icu_admission", "selected_origin": SIC_CARE_UNIT_ORIGIN,
                 "source_field": research_capability_profile("sic").care_unit_admission_proxy.model_dump(mode="json"),
                 "source_identity_sha256": canonical_sha256(identity)}
    receipt = {"source_identity_sha256": selection["source_identity_sha256"],
               "export_authority_sha256": "a" * 64, "manifest_sha256": "b" * 64,
               "selected_origin": SIC_CARE_UNIT_ORIGIN}
    fact = {"selection": selection, "registered_export_receipt": receipt,
            "disposition": "proposed_not_approved", "execution_contract_available": True}
    review = SimpleNamespace(approval_allowed=True, facts={"database_time_origin_review": fact})
    preferences = {"data_constraints": json.dumps({"source_identity": identity,
                                                  "database_time_origin_selection": selection})}
    return review, preferences, receipt


def test_candidate_preparation_rechecks_same_registered_export(monkeypatch):
    review, preferences, receipt = _candidate()
    monkeypatch.setattr(authority, "attest_registered_export_time_origin",
                        lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: receipt))
    axis = owner._reviewed_candidate_source_time_axis(
        review=review, preferences=preferences, export_path="/synthetic/registered-export",
    )
    assert axis.source_identity_sha256 == receipt["source_identity_sha256"]
    assert axis.manifest_sha256 == receipt["manifest_sha256"]


@pytest.mark.parametrize("change", ["proposed", "not_approvable", "contract_unavailable", "context_drift", "export_drift"])
def test_candidate_preparation_rejects_unselected_unapproved_or_drifted_inputs(monkeypatch, change):
    review, preferences, receipt = _candidate()
    current = copy.deepcopy(receipt)
    fact = review.facts["database_time_origin_review"]
    if change == "proposed":
        fact["disposition"] = "proposed_not_selected"
    elif change == "not_approvable":
        review.approval_allowed = False
    elif change == "contract_unavailable":
        fact["execution_contract_available"] = False
    elif change == "context_drift":
        constraints = json.loads(preferences["data_constraints"])
        constraints["source_identity"]["registration_id"] = "other"
        preferences["data_constraints"] = json.dumps(constraints)
    else:
        current["manifest_sha256"] = "c" * 64
    monkeypatch.setattr(authority, "attest_registered_export_time_origin",
                        lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: current))
    with pytest.raises(owner.ResearchPipelineRunError, match="data preparation was not started"):
        owner._reviewed_candidate_source_time_axis(
            review=review, preferences=preferences, export_path="/synthetic/registered-export",
        )
