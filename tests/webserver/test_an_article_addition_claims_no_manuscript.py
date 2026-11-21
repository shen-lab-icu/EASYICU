"""An article addition reports analyses only; it never claims a manuscript (S1799).

UX rc_d45 age leg: after 完善为论文级分析 completed, the card said
「论文稿中有部分数字与分析结果不一致」 and Copilot asked for an interpretation
review, although an addition never writes a manuscript.
"""
from types import SimpleNamespace

import pytest

from tests.support import pytest_state  # noqa: F401
from tests.support.result_append import completed_source
from easyicu.webserver import research_append_run as owner, study_contexts
from easyicu.webserver.research_append_source import read_json, seal_completed_source
from easyicu.webserver.run_record import RunGate


@pytest.fixture
def addition(tmp_path, monkeypatch):
    wrapper, run, _, _, baseline = completed_source(tmp_path)
    source = seal_completed_source(wrapper=wrapper, run_dir=run, study={"id": "study"},
        baseline=baseline.model_dump(mode="json"), prepared_package_binding={"synthetic": True})
    child = owner.propose_append_run(source=source, kind="article", wrapper=wrapper.parent / "run_article",
                                     study={"id": "study"}, registered_source_changed=False)
    monkeypatch.setattr(study_contexts, "get_context", lambda _: {"id": "study"})
    return child


def test_a_completed_article_addition_has_no_manuscript_to_fail(addition):
    owner.approve_append_run(addition.wrapper, decision="approved", reviewer="synthetic")
    gate = read_json(addition.wrapper / "quality_gate.json")["gate"]

    run_gate = RunGate.from_payload(gate)

    assert gate["status"] == "analysis_only"
    assert run_gate.manuscript_outcome.status == "not_requested"
    assert run_gate.manuscript_outcome.reason_code is None
    assert run_gate.analysis_outcome.status == "completed"


def test_a_pending_article_addition_still_awaits_its_plan_review(addition):
    gate = read_json(addition.wrapper / "quality_gate.json")["gate"]

    assert gate["reason"] == "human_plan_review_required" and "manuscript_outcome" not in gate
