"""Host-owned study readings reach the conversation without gaining write authority."""
from __future__ import annotations

import pytest

from easyicu.webserver import study_contexts
from easyicu.webserver.causal_trial_reading import causal_trial_reading
from easyicu.webserver.study_family_reading import study_family_reading
from easyicu.webserver.pi_copilot.projections import project_study_context, project_study_setup_receipt


@pytest.mark.parametrize("question, field, reading", [
    ("Is lactate associated with mortality?", "study_family_reading",
     study_family_reading("Is lactate associated with mortality?").record()),
    ("What proportion of ICU stays have mortality?", "study_family_reading",
     study_family_reading("What proportion of ICU stays have mortality?").record()),
    ("time zero at hour 6, start norepinephrine within 24 h versus not start, 28-day mortality risk",
     "causal_trial_reading", causal_trial_reading("time zero at hour 6, start norepinephrine within 24 h versus not start, 28-day mortality risk").record()),
])
def test_reading_and_its_original_evidence_reach_the_setup_receipt(question, field, reading):
    study = {"id": "study-reading", "revision": 2, "question": question, field: reading}
    assert project_study_context(study)[field] == reading
    assert project_study_setup_receipt(study).configuration[field] == reading


def test_variance_conflict_is_shown_without_changing_the_design():
    basis = {"variance_estimator": "none_counts_only", "conflict": "counts_only_not_requested"}
    study = {"id": "study-reading", "revision": 2, "question": "What proportion of stays have mortality?",
             "analysis_design": {"analysis_family": "descriptive_epidemiology", "variance_estimator": "none_counts_only"},
             "design_variance_basis": basis}
    receipt = project_study_setup_receipt(study)
    assert receipt.configuration["design_variance_basis"] == basis
    assert receipt.configuration["analysis_design"]["variance_estimator"] == "none_counts_only"


def test_public_client_still_cannot_write_a_family_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "study.json")
    record = study_family_reading("Is lactate associated with mortality?").record()
    with pytest.raises(study_contexts.StudyContextError, match="study_family_reading_server_owned"):
        study_contexts.upsert_context({"id": "study-reading", "study_family_reading": record})
