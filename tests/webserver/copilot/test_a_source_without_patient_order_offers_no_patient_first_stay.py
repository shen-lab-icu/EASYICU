"""A source that cannot order one patient's ICU stays offers no per-patient first stay.

eICU orders units only within one hospitalization, and HiRID links no
patient, so neither can select each patient's first ICU stay. A study asking
for it on them was saved, planned on the raw source, and then refused by the
population review and the extraction in turn. It is now refused before
anything is saved, with the populations the source can select instead.
"""

from __future__ import annotations

import copy

import pytest

from easyicu.webserver import study_contexts
from easyicu.webserver.pi_copilot import cohort_eligibility
from easyicu.webserver.pi_copilot.contracts import AuthorityBinding, PiSessionRecord, ToolExecutionContext
from easyicu.webserver.pi_copilot.study_context_update import update_study_context
from easyicu.webserver.pi_copilot.workflow import build_research_workflow_snapshot
from easyicu.webserver.raw_source_authority import patient_first_icu_stay_substitutes

MESSAGE = "只研究每位患者的首次 ICU 入住（按患者，不按住院）"


@pytest.fixture
def make_context(tmp_path, monkeypatch):
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "studies.json")

    def create(database, cohort=None):
        study = study_contexts.upsert_context({
            "title": "Synthetic study",
            "question": "Is lactate associated with hospital mortality?",
            "data_source": {"path": str(tmp_path / database), "database": database},
            "cohort": dict(cohort or {}),
        })
        session = PiSessionRecord(
            session_id="synthetic",
            binding=AuthorityBinding(study_context_id=study["id"], study_revision=study["revision"]),
        )
        return study, ToolExecutionContext(session=session, user_message=MESSAGE, allowed_actions=("configure",))

    return create


@pytest.mark.parametrize(("database", "substitutes"), [
    ("eicu_demo", ["first_icu_stay_per_hospitalization", "all_icu_stays"]),
    ("eicu", ["first_icu_stay_per_hospitalization", "all_icu_stays"]),
    ("hirid", ["all_icu_stays"]),
])
@pytest.mark.parametrize("proposal", [{"preset": "adult_first"}, {"exclude_readmissions": True}])
def test_a_patient_first_stay_is_refused_before_saving(make_context, database, substitutes, proposal):
    study, context = make_context(database, {"first_icu_stay_per_hospitalization": True}
                                  if "eicu" in database else {})
    before = copy.deepcopy(study_contexts.get_context(study["id"]))
    result = update_study_context(context, {"cohort": proposal})
    assert (result["status"], result["code"]) == ("blocked", "study_first_icu_stay_per_patient_unavailable")
    assert result["details"] == {"field": "cohort", "database": database, "substitutes": substitutes}
    assert study_contexts.get_context(study["id"]) == before
    assert context.grant.consume_once("configure") == "granted"


@pytest.mark.parametrize("database", ["miiv", "mimic", "aumc", "miiv_demo", "sic"])
def test_a_source_that_can_or_is_not_declared_keeps_the_patient_first_stay(make_context, database):
    assert patient_first_icu_stay_substitutes(database) is None
    study, context = make_context(database)
    result = update_study_context(context, {"cohort": {"preset": "adult_first"}})
    assert result["code"] != "study_first_icu_stay_per_patient_unavailable"


def test_eligibility_choices_and_workflow_follow_the_source(make_context):
    eicu, _ = make_context("eicu_demo")
    offered = {option["id"] for option in cohort_eligibility.selection_options_for_study(eicu)}
    assert offered and not offered & {"first_admission_only", "adults_first_admission", "adults_first_admission_min_stay"}
    snapshot = build_research_workflow_snapshot(study=eicu, active_export_present=True, active_job=None, latest_run=None)
    assert snapshot.patient_first_stay_substitutes == ["first_icu_stay_per_hospitalization", "all_icu_stays"]

    miiv, _ = make_context("miiv")
    offered = {option["id"] for option in cohort_eligibility.selection_options_for_study(miiv)}
    assert "adults_first_admission" in offered
    snapshot = build_research_workflow_snapshot(study=miiv, active_export_present=True, active_job=None, latest_run=None)
    assert snapshot.patient_first_stay_substitutes is None
